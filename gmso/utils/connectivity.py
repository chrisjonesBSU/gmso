"""Module supporting various connectivity methods and operations."""

import itertools
import re
from itertools import combinations
from typing import TYPE_CHECKING, List

import networkx as nx
from boltons.setutils import IndexedSet
from networkx.algorithms import shortest_path_length

if TYPE_CHECKING:
    from gmso import Topology
    from gmso.core.atom import Site
    from gmso.core.bond import Bond

from gmso.core.angle import Angle
from gmso.core.dihedral import Dihedral
from gmso.core.improper import Improper
from gmso.core.virtual_site import VirtualSite
from gmso.exceptions import MissingParameterError

CONNS = {"angle": Angle, "dihedral": Dihedral, "improper": Improper}

# EDGES is retained as a public constant; it is no longer used internally
# for detection (which now uses direct adjacency enumeration) but may be
# referenced by external callers.
EDGES = {
    "angle": ((0, 1),),
    "dihedral": ((0, 1), (1, 2)),
    "improper": ((0, 1), (0, 2), (1, 2)),
}


def identify_connections(top, index_only=False):
    """Identify all possible connections within a topology.

    Parameters
    ----------
    top : gmso.Topology
        The gmso topology for which to identify connections for.
    index_only : bool, default=False
        If True, return integer site indices that would form the connections
        rather than adding the connections to the topology.

    Notes
    -----
    Connections are detected by direct adjacency enumeration over the
    topology's bond graph (built with integer site indices as nodes).
    This replaces the previous approach of VF2 subgraph isomorphism on
    the line graph of the bond graph.

    The key insight is that the patterns being searched for — angles,
    dihedrals, impropers — are simple enough that they can be enumerated
    in O(E * max_degree) time by walking the adjacency structure directly:

    - Angle   (a-b-c): for each node b, enumerate pairs of its neighbors.
    - Dihedral (a-b-c-d): for each edge (b,c), enumerate (neighbor of b) x
                          (neighbor of c), excluding the b-c bond itself.
    - Improper (central; b1,b2,b3): for each node with degree >= 3,
                          enumerate all combinations of 3 neighbors.

    Site objects are only touched at two boundaries:
      - Entry: building the site_index_map (site -> int).
      - Exit:  _add_connections resolving int indices back to Site objects.
    Everything in between operates on plain Python ints.
    """
    # Build site -> index map once up front. top._sites is an IndexedSet
    # (O(1) .index()), but constructing this dict means the bond loop below
    # pays a single dict lookup per member rather than a method call.
    site_index_map = {site: i for i, site in enumerate(top.sites)}

    # Build an integer-node adjacency dict directly from bonds.
    # Using dict[int, set[int]] rather than nx.Graph avoids all networkx
    # per-node overhead during the enumeration loops.
    adj: dict[int, set[int]] = {}
    for b in top.bonds:
        i = site_index_map[b.connection_members[0]]
        j = site_index_map[b.connection_members[1]]
        if i not in adj:
            adj[i] = set()
        if j not in adj:
            adj[j] = set()
        adj[i].add(j)
        adj[j].add(i)

    angle_matches    = _enumerate_angles(adj)
    dihedral_matches = _enumerate_dihedrals(adj)
    improper_matches = _enumerate_impropers(adj)

    if not index_only:
        for conn_matches, conn_type in zip(
            (angle_matches, dihedral_matches, improper_matches),
            ("angle", "dihedral", "improper"),
        ):
            if conn_matches:
                _add_connections(top, conn_matches, conn_type=conn_type)
    else:
        return {
            "angles": angle_matches,
            "dihedrals": dihedral_matches,
            "impropers": improper_matches,
        }

    return top


def _add_connections(top, matches, conn_type):
    """Add connections to the topology."""
    for sorted_conn in matches:
        cmembers = [top.sites[idx] for idx in sorted_conn]
        bonds = list()
        for i, j in CONNS[conn_type].connectivity:
            bond = (cmembers[i], cmembers[j])
            key = frozenset([bond, tuple(reversed(bond))])
            bonds.append(top._unique_connections[key])
        to_add_conn = CONNS[conn_type](connection_members=cmembers, bonds=tuple(bonds))
        top.add_connection(to_add_conn, update_types=False)


def _enumerate_angles(adj):
    """Enumerate all angles by direct adjacency traversal.

    An angle is any triple (a, b, c) where a and c are both bonded to b.
    For each node b with at least 2 neighbors, we enumerate all unordered
    pairs of those neighbors.

    Canonicalisation: smaller terminal index first, so (a, b, c) with a < c.
    This is imposed at construction time so the set handles deduplication.

    Parameters
    ----------
    adj : dict[int, set[int]]
        Adjacency dict of the integer-node bond graph.

    Returns
    -------
    list of tuple[int, int, int]
        Sorted list of (end0, middle, end1) triples with end0 < end1,
        ordered by (middle, end0, end1).
    """
    matches = set()
    for b, neighbors in adj.items():
        if len(neighbors) < 2:
            continue
        for a, c in combinations(neighbors, 2):
            # Canonicalise: smaller end first.
            if a < c:
                matches.add((a, b, c))
            else:
                matches.add((c, b, a))
    return sorted(matches, key=lambda x: (x[1], x[0], x[2]))


def _enumerate_dihedrals(adj):
    """Enumerate all dihedrals by direct adjacency traversal.

    A dihedral is any quadruple (a, b, c, d) where a-b, b-c, and c-d are
    all bonds and a != c, b != d. For each bond (b, c), we enumerate all
    valid (a, d) pairs where a is a neighbor of b other than c, and d is a
    neighbor of c other than b.

    Each bond is only visited once (enforced by c > b) to avoid emitting
    both (a,b,c,d) and its mirror (d,c,b,a) before canonicalisation.
    Canonicalisation (smaller terminal first) then handles any remaining
    orientation ambiguity.

    Parameters
    ----------
    adj : dict[int, set[int]]
        Adjacency dict of the integer-node bond graph.

    Returns
    -------
    list of tuple[int, int, int, int]
        Sorted list of (a, b, c, d) quadruples with a < d (canonical form),
        ordered by (b, c, a, d).
    """
    matches = set()
    for b, b_neighbors in adj.items():
        for c in b_neighbors:
            if c <= b:
                # Process each bond once; the c > b side handles both
                # orientations via canonicalisation below.
                continue
            for a in b_neighbors:
                if a == c:
                    continue
                for d in adj[c]:
                    if d == b:
                        continue
                    # Canonicalise: smaller terminal first.
                    if a < d:
                        matches.add((a, b, c, d))
                    else:
                        matches.add((d, c, b, a))
    return sorted(matches, key=lambda x: (x[1], x[2], x[0], x[3]))


def _enumerate_impropers(adj):
    """Enumerate all impropers by direct adjacency traversal.

    An improper is any quadruple (central, b1, b2, b3) where central is
    bonded to all three of b1, b2, b3. For each node with degree >= 3,
    we enumerate all combinations of 3 neighbors as the branch atoms.

    Canonicalisation: branches are sorted ascending. Central node identity
    is unambiguous so no further orientation handling is needed.

    Parameters
    ----------
    adj : dict[int, set[int]]
        Adjacency dict of the integer-node bond graph.

    Returns
    -------
    list of tuple[int, int, int, int]
        Sorted list of (central, b1, b2, b3) with b1 < b2 < b3,
        ordered by (central, b1, b2, b3).
    """
    matches = set()
    for central, neighbors in adj.items():
        if len(neighbors) < 3:
            continue
        for trio in combinations(neighbors, 3):
            b1, b2, b3 = sorted(trio)
            matches.add((central, b1, b2, b3))
    return sorted(matches, key=lambda x: (x[0], x[1], x[2], x[3]))


# ---------------------------------------------------------------------------
# The functions below (_detect_connections and the subgraph formatters) are
# retained for backwards compatibility only. They are no longer called by
# identify_connections. External code that calls _detect_connections directly
# can continue to do so, but should migrate to the enumerate functions above.
# ---------------------------------------------------------------------------

def _detect_connections(compound_line_graph, top=None, type_="angle"):
    """Detect connections via VF2 subgraph isomorphism on the line graph.

    .. deprecated::
        This function is no longer used by identify_connections, which now
        calls _enumerate_angles / _enumerate_dihedrals / _enumerate_impropers
        directly. Retained for external callers only.

    Parameters
    ----------
    compound_line_graph : nx.Graph
        Line graph of the bond graph.
    top : gmso.Topology or None
        No longer used; retained for signature compatibility.
    type_ : str
        One of 'angle', 'dihedral', 'improper'.
    """
    _CONNECTION_GRAPHS = {}
    for _type, _edges in EDGES.items():
        _g = nx.Graph()
        for _edge in _edges:
            _g.add_edge(*_edge)
        _CONNECTION_GRAPHS[_type] = _g

    _SORT_KEYS = {
        "angle":    lambda a: (a[1], a[0], a[2]),
        "dihedral": lambda d: (d[1], d[2], d[0], d[3]),
        "improper": lambda i: (i[0], i[1], i[2], i[3]),
    }

    matcher = nx.algorithms.isomorphism.GraphMatcher(
        compound_line_graph, _CONNECTION_GRAPHS[type_]
    )
    formatter_fn = {
        "angle":    _format_subgraph_angle,
        "dihedral": _format_subgraph_dihedral,
        "improper": _format_subgraph_improper,
    }[type_]

    conn_matches = IndexedSet()
    for m in matcher.subgraph_isomorphisms_iter():
        match = formatter_fn(m)
        if match is None:
            continue
        if type_ in ("angle", "dihedral"):
            if match[0] > match[-1]:
                match = match[::-1]
        elif type_ == "improper":
            match = (match[0],) + tuple(sorted(match[1:]))
        conn_matches.add(match)

    return sorted(conn_matches, key=_SORT_KEYS[type_])


def _get_sorted_by_n_connections(m):
    """Return nodes sorted by degree for a VF2 match dict. Retained for
    backwards compatibility with _detect_connections."""
    small = nx.Graph()
    for k in m:
        small.add_edge(k[0], k[1])
    return sorted(small.adj, key=lambda x: len(small[x])), small


def _format_subgraph_angle(m):
    """Format an angle subgraph match dict. Retained for backwards
    compatibility with _detect_connections."""
    sort_by_n_connections, _ = _get_sorted_by_n_connections(m)
    end0, end1 = sorted([sort_by_n_connections[0], sort_by_n_connections[1]])
    middle = sort_by_n_connections[2]
    return (end0, middle, end1)


def _format_subgraph_dihedral(m):
    """Format a dihedral subgraph match dict. Retained for backwards
    compatibility with _detect_connections."""
    sort_by_n_connections, small = _get_sorted_by_n_connections(m)
    start = sort_by_n_connections[0]
    if sort_by_n_connections[2] in small.neighbors(start):
        mid1 = sort_by_n_connections[2]
        mid2 = sort_by_n_connections[3]
    else:
        mid1 = sort_by_n_connections[3]
        mid2 = sort_by_n_connections[2]
    end = sort_by_n_connections[1]
    return (start, mid1, mid2, end)


def _format_subgraph_improper(m):
    """Format an improper subgraph match dict. Retained for backwards
    compatibility with _detect_connections."""
    sort_by_n_connections, _ = _get_sorted_by_n_connections(m)
    if len(sort_by_n_connections) == 4:
        central = sort_by_n_connections[3]
        branch1, branch2, branch3 = sorted(sort_by_n_connections[:3])
        return (central, branch1, branch2, branch3)
    return None


def _trim_duplicates(all_matches):
    """Remove redundant sub-graph matches.

    .. deprecated::
        No longer called by the main pipeline. Retained for external callers.
    """
    trimmed_list = IndexedSet()
    for match in all_matches:
        if match and match not in trimmed_list and match[::-1] not in trimmed_list:
            trimmed_list.add(match)
    return trimmed_list


# ---------------------------------------------------------------------------
# Everything below this line is unchanged from the original module.
# ---------------------------------------------------------------------------

def generate_pairs_lists(
    top, molecule=None, sort_key=None, refer_from_scaling_factor=False
):
    """Generate all the pairs lists of the topology or molecular of topology.

    Parameters
    ----------
    top : gmso.Topology
        The Topology where we want to generate the pairs lists from.
    molecule : molecule namedtuple, optional, default=None
        Generate only pairs list of a particular molecule.
    sort_key : function, optional, default=None
        Function used as key for sorting of site pairs. If None is provided
        will used topology.get_index
    refer_from_scaling_factor : bool, optional, default=False
        If True, only generate pair lists of pairs that have a non-zero scaling
        factor value.

    Returns
    -------
    pairs_lists: dict of list
        {"pairs12": pairs12, "pairs13": pairs13, "pairs14": pairs14}

    NOTE: This method assume that the topology has already been loaded with
    angles and dihedrals (through top.identify_connections()). In addition,
    if the refer_from_scaling_factor is True, this method will only generate
    pairs when the corresponding scaling factor is not 0.
    """
    from gmso.external import to_networkx
    from gmso.parameterization.molecule_utils import (
        molecule_angles,
        molecule_bonds,
        molecule_dihedrals,
    )

    nb_scalings, coulombic_scalings = top.scaling_factors

    if sort_key is None:
        sort_key = top.get_index

    graph = to_networkx(top, parse_angles=False, parse_dihedrals=False)

    pairs_dict = dict()
    if refer_from_scaling_factor:
        for i in range(3):
            if nb_scalings[i] or coulombic_scalings[i]:
                pairs_dict[f"pairs1{i + 2}"] = list()
    else:
        for i in range(3):
            pairs_dict = {f"pairs1{i + 2}": list() for i in range(3)}

    if molecule is None:
        bonds, angles, dihedrals = top.bonds, top.angles, top.dihedrals
    else:
        bonds = molecule_bonds(top, molecule)
        angles = molecule_angles(top, molecule)
        dihedrals = molecule_dihedrals(top, molecule)

    if "pairs12" in pairs_dict:
        for bond in bonds:
            pairs = sorted(bond.connection_members, key=sort_key)
            pairs_dict["pairs12"].append(pairs)

    if "pairs13" in pairs_dict:
        for angle in angles:
            pairs = sorted(
                (angle.connection_members[0], angle.connection_members[-1]),
                key=sort_key,
            )
            if (
                pairs not in pairs_dict["pairs13"]
                and shortest_path_length(graph, pairs[0], pairs[1]) == 2
            ):
                pairs_dict["pairs13"].append(pairs)

    if "pairs14" in pairs_dict:
        for dihedral in dihedrals:
            pairs = sorted(
                (
                    dihedral.connection_members[0],
                    dihedral.connection_members[-1],
                ),
                key=sort_key,
            )
            if (
                pairs not in pairs_dict["pairs14"]
                and shortest_path_length(graph, pairs[0], pairs[1]) == 3
            ):
                pairs_dict["pairs14"].append(pairs)

    for key in pairs_dict:
        pairs_dict[key] = sorted(
            pairs_dict[key],
            key=lambda pairs: (sort_key(pairs[0]), sort_key(pairs[1])),
        )

    return pairs_dict


def identify_virtual_sites(
    topology: "Topology",
    sites: List["Site"],
    bonds: List["Bond"],
    virtual_types: List[VirtualSite],
):
    """Identify virtual sites within an already typed topology based on the virtual_types.

    Parameters
    ----------
    topology : gmso.Topology
        Topology to search for parameters.
    sites : List[gmso.core.abstract_site.Site]
        Sites to use to construct subsearch of topology. Can be all sites in
        the topology, or a subset of sites.
    bonds : List[gmso.core.bonds.Bond]
        Bonds to use to construct subsearch of topology. Can be all bonds in
        the topology, or a subset of bonds.
    virtual_types : List[gmso.core.virtual_types.VirtualType]
        Virtual types, presumably from a gmso.ForceField, used to match the
        parent_atoms in the sites and bonds graph.

    Returns
    -------
    virtual_sites : List[gmso.core.virtual_site.VirtualSite]
        VirtualSite instances identified in the topology.
    """
    for site in sites:
        if not site.atom_type:
            raise MissingParameterError(site.atom_type, "atom_type")
    compound = nx.Graph()

    for b in bonds:
        compound.add_node(b.connection_members[0], identifier=b.member_types[0])
        compound.add_node(b.connection_members[1], identifier=b.member_types[1])
        compound.add_edge(b.connection_members[0], b.connection_members[1])

    virtual_sites = []
    for vtype in virtual_types.values():
        vtype_graph = _graph_from_vtype(vtype)
        matchesMap = _get_graph_isomorphism_matches(compound, vtype_graph)
        for match in matchesMap.values():
            vsite = VirtualSite(parent_sites=match.keys())
            virtual_sites.append(vsite)
            topology._add_virtual_site(vsite)

    return virtual_sites


def _get_graph_isomorphism_matches(g1, g2, match_by="identifier"):
    """g1 is a large map that is checked for g2 subgraphs in."""
    node_match = nx.algorithms.isomorphism.categorical_node_match(match_by, default="")
    graph_matcher = nx.algorithms.isomorphism.GraphMatcher(
        g1, g2, node_match=node_match
    )
    acceptedMaps = dict()
    for mapping in graph_matcher.subgraph_isomorphisms_iter():
        possibleMap = {g1id: g2id for g1id, g2id in mapping.items()}
        acceptedMaps[frozenset(possibleMap.keys())] = possibleMap

    return acceptedMaps


def _graph_from_vtype(vtype):
    """Create a graph from a virtual_type."""
    virtual_type_graph = nx.Graph()
    if vtype.member_types:
        iter_elementsStr = "member_types"
    else:
        iter_elementsStr = "member_classes"
    for i, member in enumerate(getattr(vtype, iter_elementsStr)):
        virtual_type_graph.add_node(i, identifier=member)
    for i in range(len(getattr(vtype, iter_elementsStr)) - 1):
        virtual_type_graph.add_edge(i, i + 1)

    return virtual_type_graph


def connection_identifier_to_string(identifier):
    """Take a list of [site1, site2, bond1] and reorder into a string identifier.

    Parameters
    ----------
    identifier : tuple, list
        The identifier for a given connection with a list of sites and bonds.
        For example, a dihedral would look like:
        combination = dihedral.connection_members + dihedral.bonds

    Returns
    -------
    pattern : str
        The identifying pattern for the list of sites. An improper might look like:
        `central_atom-atom2-atom3=atom4`
        where the combination was:
        ["central_atom", "atom2", "atom3", "atom4", "-", "-", "="]
    """
    bonds_cutoff = len(identifier) // 2
    sites = identifier[: bonds_cutoff + 1]
    bonds = identifier[bonds_cutoff + 1 :]
    pattern = sites[0]
    for b, sit in zip(bonds, sites[1:]):
        pattern += b + sit
    return pattern


def yield_connection_identifiers(identifier):
    """Yield all possible bond identifiers from a tuple or string identifier."""
    n_sites = len(identifier) // 2 + 1
    if isinstance(identifier, str):
        bond_tokens = r"([\=\~\-\#\:])"
        identifier = re.split(bond_tokens, identifier)
        identifier = identifier[::2] + identifier[1::2]
    site_identifiers = identifier[:n_sites]
    bond_identifiers = identifier[n_sites:]
    choices = [(site_identifier, "*") for site_identifier in site_identifiers]
    choices += [(val, "~") for val in bond_identifiers]
    return itertools.product(*choices)
