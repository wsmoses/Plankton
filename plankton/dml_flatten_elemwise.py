""" DaCeML optimization that flattens elementwise kernels. """
import dace
import numpy as np
from dace import data as dt
from dace.transformation import helpers as xfh


def flatten_elementwise_kernels_sdfg(sdfg: dace.SDFG):
    from dace.data import _prod
    from dace.subsets import Range

    def is_c_contiguous(desc):
        cstrides = [_prod(desc.shape[i + 1:]) for i in range(len(desc.shape))]
        return tuple(desc.strides) == tuple(cstrides)

    def match(sdfg, state, mapentry):
        if len(mapentry.params) == 1:
            return None
        numel = mapentry.range.num_elements()
        mapexit = state.exit_node(mapentry)

        map_memlet = dace.Memlet(data='unused',
                                 subset=','.join(mapentry.params))
        rng = map_memlet.subset

        # Get all memlets within map
        memlets = []
        for in_edge in state.out_edges(mapentry):
            src_node = state.memlet_path(in_edge)[0].src
            if not isinstance(src_node, dace.nodes.AccessNode):
                return None
            desc = sdfg.arrays[src_node.data]

            # Ignore input scalars
            if isinstance(desc, dt.Scalar) or desc.total_size == 1:
                continue

            if desc.total_size != numel or not is_c_contiguous(desc):
                return None

            # Check that memlet matches map parameters
            if in_edge.data.subset != rng:
                return None

            memlets.append(in_edge)

        for out_edge in state.in_edges(mapexit):
            dst_node = state.memlet_path(out_edge)[-1].dst
            if not isinstance(dst_node, dace.nodes.AccessNode):
                return None
            desc = sdfg.arrays[dst_node.data]
            if desc.total_size != numel or not is_c_contiguous(desc):
                return None

            # Check that memlet matches map parameters
            if out_edge.data.subset != rng:
                return None

            memlets.append(out_edge)

        return memlets

    for mapentry, state in sdfg.all_nodes_recursive():
        if not isinstance(mapentry, dace.nodes.MapEntry):
            continue
        nsdfg = state.parent
        m = match(nsdfg, state, mapentry)
        if m is None:
            continue

        # Flatten map
        elem = dace.symbol('__elem')
        mapentry.params = ['__elem']
        numel = mapentry.range.num_elements()
        mapentry.range = Range([(0, numel - 1, 1)])

        for edge in m:
            mtree = state.memlet_tree(edge)

            # Add view
            outer_edge = mtree.parent.edge

            # Skip if view was already added
            if outer_edge.dst is mapentry:
                if type(outer_edge.src.desc(sdfg)) is dt.View:
                    continue
            else:
                if type(outer_edge.dst.desc(sdfg)) is dt.View:
                    continue

            desc = nsdfg.arrays[outer_edge.data.data]
            view, _ = nsdfg.add_view('__v' + outer_edge.data.data, [numel],
                                     desc.dtype,
                                     desc.storage,
                                     find_new_name=True)
            vnode = state.add_access(view)

            # Modify internal memlets
            for mnode in mtree:
                e = mnode
                e.data.subset = Range([(elem, elem, 1)])
                e.data.data = view

            # Connect through view
            state.remove_edge(outer_edge)
            if outer_edge.dst is mapentry:
                state.add_edge(outer_edge.src, outer_edge.src_conn, vnode,
                               None, dace.Memlet(outer_edge.data.data))
                state.add_edge(vnode, None, outer_edge.dst,
                               outer_edge.dst_conn, dace.Memlet(view))
            else:
                state.add_edge(outer_edge.src, outer_edge.src_conn, vnode,
                               None, dace.Memlet(view))
                state.add_edge(vnode, None, outer_edge.dst,
                               outer_edge.dst_conn,
                               dace.Memlet(outer_edge.data.data))


def flatten_elementwise_autodiff(mod):
    def spec(fwd, bwd):
        flatten_elementwise_kernels_sdfg(fwd)
        flatten_elementwise_kernels_sdfg(bwd)

    mod.append_post_autodiff_hook("flatten", spec)


def flatten_elementwise_autodiff_fwd(mod):
    def spec(fwd, bwd):
        flatten_elementwise_kernels_sdfg(fwd)

    mod.append_post_autodiff_hook("flatten", spec)


def flatten_elementwise_onnx(mod):
    def spec(module):
        flatten_elementwise_kernels_sdfg(module.sdfg)

    mod.append_post_onnx_hook("flatten", spec)
