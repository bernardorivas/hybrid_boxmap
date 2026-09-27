"""
Hybrid Dynamics - A Python library for hybrid dynamical systems analysis.

This library provides tools for:
- Hybrid system simulation with event detection
- Enhanced trajectory tracking with hybrid time domains
- Box map computation for global dynamics approximation
- Morse graph analysis of recurrent components
- Visualization of hybrid systems, box maps, and analysis results
"""

import logging


_package_logger = logging.getLogger(__name__)
if not any(
    isinstance(handler, logging.NullHandler) for handler in _package_logger.handlers
):
    _package_logger.addHandler(logging.NullHandler())

# Global configuration
# Cubical grid components
from .src.box import Box
from .src.config import config, configure_logging, get_logger
from .src.grid import Grid
from .src.hybrid_boxmap import HybridBoxMap
from .src.cmgdb_suspension_boxmap import (
    CMGDBSuspensionBoxMap,
    SINGLE_HANDLE_BRIDGE_ALGORITHM,
    SINGLE_HANDLE_BRIDGE_ASSUMPTIONS,
    SamplingFailure,
    SingleHandleBridgeDiagnostic,
    SingleHandleBridgeProbe,
    SourceBoxMapDiagnostic,
    SuspensionAtlasCharts,
    SuspensionBoxMapDiagnostics,
    SuspensionStratum,
    TaggedRectangle,
    build_cmgdb_atlas_model,
)
from .src.hybrid_system import HybridSystem
from .src.hybrid_time import HybridTime, HybridTimeInterval
from .src.hybrid_trajectory import HybridTrajectory, TrajectorySegment
from .src.implicit_phase_scc import (
    FullSCCReconstruction,
    PhaseDescriptor,
    PhaseGadgetDescriptor,
    PhasePathDescriptor,
    VirtualPhaseNode,
    build_base_macro_graph,
    reconstruct_from_base_descriptors,
    reconstruct_full_scc_condensation,
)
from .src.fixed_time_suspension_grid import (
    FixedTimeSuspensionGridResult,
    GridResetHandle,
    GridSuspensionIngredients,
    SuspensionCemeteryCell,
    compute_fixed_time_suspension_grid,
)
from .src.fixed_time_relation_audit import (
    AuditedCell,
    AugmentedQuotientIncidence,
    CellSetConnectivity,
    EndpointCoverageAudit,
    EndpointCoverageWitness,
    EndpointEvaluationFailure,
    EndpointProbe,
    RelationAuditFailure,
    RelationConnectivityAudit,
    audit_cell_sets_connectivity,
    audit_dense_base_endpoints,
    audit_endpoint_probes,
    audit_relation_image_connectivity,
)
from .src.hybrid_morse_plot import (
    CMGDB_MORSE_PALETTE,
    SCIENTIFIC_MORSE_PALETTE,
    HybridMorseComponent,
    HybridMorsePlot,
    PlotHybridMorseSets,
    hybrid_morse_components,
    hybrid_morse_hasse,
    plot_hybrid_morse_sets,
    save_hybrid_morse_figure,
)
from .src.atlas_morse_plot import (
    ATLAS_MORSE_PLOT_SCHEMA,
    PHYSICAL_CONLEY_FINITE_RELATION_AUDIT_SCHEMA,
    AtlasFiniteRelationIndexAnnotations,
    AtlasHybridMorseComponent,
    AtlasHybridMorsePlot,
    AtlasMorseBox,
    AtlasMorseNode,
    AtlasMorsePlotData,
    atlas_morse_components,
    atlas_morse_hasse,
    atlas_morse_plot_data_payload,
    extract_atlas_morse_plot_data,
    load_atlas_morse_plot_data,
    load_atlas_finite_relation_index_annotations,
    plot_atlas_hybrid_morse_sets,
    save_atlas_morse_plot_data,
)

# Graph analysis functions
from .src.morse_graph import create_morse_graph

# MultiGrid framework
from .src.multigrid import MultiGrid, MultiGridBoxMap

# Visualization functions
from .src.plot_utils import (
    HybridPlotter,
    plot_morse_graph_viz,
    plot_morse_sets_on_grid,
    plot_morse_sets_on_grid_fast,
    plot_morse_sets_with_roa,
    plot_morse_sets_with_roa_fast,
    visualize_box_map,
    visualize_box_map_entry,
    visualize_flow_map,
)

# Region of attraction analysis
from .src.roa_utils import (
    analyze_roa_coverage,
    compute_regions_of_attraction,
    compute_roa,
)
from .src.sampled_suspension import (
    AugmentedCell,
    BaseSuspensionSample,
    BaseCell,
    HandleSuspensionSample,
    PhaseCell,
    PhaseOnlyRecurrentComponentError,
    RecurrentMorseCollapseDiagnostic,
    SuspensionSample,
    assert_recurrent_morse_equivalence,
    build_augmented_outer_graph,
    collapse_phase_paths,
    crossing_completed_state,
    diagnose_recurrent_morse_collapse,
    locate_augmented_cells,
    sample_suspension_trajectory,
    simulate_crossing_completed_state,
    simulate_suspension_endpoint,
)
from .src.suspension_complex import (
    CMGDBCellRelationPayload,
    CMGDBRelativeHomologyPayload,
    CellularAttachmentMap,
    CellularChainMap,
    CellularMapBetweenComplexes,
    CellularResetMap,
    CrossComplexAcyclicCarrier,
    CubicalCell,
    CubicalGridComplex,
    CubicalHyperplaneAttachmentAudit,
    DoubleMappingCylinderComplex,
    DoubleMappingCylinderHandle,
    FiniteCellComplex,
    FixedTimeCarrier,
    FixedTimeCellRelation,
    GuardPrismCell,
    PhaseCellRegistry,
    PhaseSliceCell,
    RelativeCellPair,
    ResetHandle,
    SampledSuspensionCellAdapter,
    SparseCubicalGridComplex,
    SuspensionBaseCell,
    SuspensionCellComplex,
    audit_cubical_hyperplane_attachment,
    build_cubical_grid_complex,
)
from .src.atlas_conley import (
    AffineBoundaryEmbedding2D,
    AtlasGoodCoverError,
    AtlasMappingCylinderConleyPreparation,
    AtlasMappingCylinderRegistry,
    AtlasMappingCylinderRelativePair,
    AtlasMappingCylinderTopCell,
    AtlasNerveSimplex,
    AtlasPhysicalConleyPreparation2D,
    AtlasQuotientNerveComplex2D,
    AtlasRectangleCell2D,
    AtlasRelativeIndexPair2D,
    AtlasResetGluing2D,
    AtlasSeamSubcomplexAudit2D,
    PhysicalConleyCertificationError,
    QuotientIntersectionAudit2D,
    QuotientRectangleRepresentative2D,
    RawAtlasCellCover2D,
    atlas_cells_from_morse_graph,
    atlas_cells_from_phase_space,
    prepare_atlas_physical_conley_2d,
    prepare_atlas_mapping_cylinder_conley,
    prepare_atlas_relation_conley_2d,
)

from .src.suspension_grid import (
    DyadicBaseWindow,
    GeneratorKey,
    GuardResetSpec,
    LocatedPoints,
    SuspensionGrid,
    SuspensionGridCheck,
    UnsupportedSuspensionGridError,
    build_suspension_grid,
    check_suspension_grid,
)
from .src.suspension_grid_relation import (
    EndpointBatch,
    SuspensionFlow,
    SuspensionGridProblem,
    SuspensionGridRelation,
    SuspensionMorseGraph,
    SuspensionPath,
    atom_set_components,
    audit_suspension_grid_endpoints,
    compute_suspension_grid_relation,
    compute_suspension_morse_graph,
    piece_evaluation_offsets,
    relation_image_connectivity,
)
from .src.suspension_grid_conley import (
    SuspensionGridConleyResult,
    compute_suspension_grid_conley_index,
    piece_rectangles,
    suspension_grid_gluing,
)
from .src.suspension_grid_plot import suspension_grid_morse_plot_data

__version__ = "0.2.0"
__author__ = "Your Name/Team"

__all__ = [
    # Paper suspension grid Xi_n (def:suspension-grid) and its sampled map
    "DyadicBaseWindow",
    "GeneratorKey",
    "GuardResetSpec",
    "LocatedPoints",
    "SuspensionGrid",
    "SuspensionGridCheck",
    "UnsupportedSuspensionGridError",
    "build_suspension_grid",
    "check_suspension_grid",
    "EndpointBatch",
    "SuspensionFlow",
    "SuspensionGridProblem",
    "SuspensionGridRelation",
    "SuspensionMorseGraph",
    "SuspensionPath",
    "atom_set_components",
    "audit_suspension_grid_endpoints",
    "compute_suspension_grid_relation",
    "compute_suspension_morse_graph",
    "piece_evaluation_offsets",
    "relation_image_connectivity",
    "SuspensionGridConleyResult",
    "compute_suspension_grid_conley_index",
    "piece_rectangles",
    "suspension_grid_gluing",
    "suspension_grid_morse_plot_data",
    # Core
    "HybridTime",
    "HybridTimeInterval",
    "HybridTrajectory",
    "TrajectorySegment",
    "HybridSystem",
    "HybridBoxMap",
    "CMGDBSuspensionBoxMap",
    "SINGLE_HANDLE_BRIDGE_ALGORITHM",
    "SINGLE_HANDLE_BRIDGE_ASSUMPTIONS",
    "SuspensionAtlasCharts",
    "SuspensionStratum",
    "SamplingFailure",
    "SingleHandleBridgeDiagnostic",
    "SingleHandleBridgeProbe",
    "SourceBoxMapDiagnostic",
    "SuspensionBoxMapDiagnostics",
    "TaggedRectangle",
    "build_cmgdb_atlas_model",
    # Cubical
    "Box",
    "Grid",
    # Graph
    "create_morse_graph",
    # ROA
    "compute_roa",
    "compute_regions_of_attraction",
    "analyze_roa_coverage",
    # MultiGrid
    "MultiGrid",
    "MultiGridBoxMap",
    # Plotting
    "HybridPlotter",
    "visualize_flow_map",
    "visualize_box_map",
    "visualize_box_map_entry",
    "plot_morse_sets_on_grid",
    "plot_morse_sets_with_roa",
    "plot_morse_sets_on_grid_fast",
    "plot_morse_sets_with_roa_fast",
    "plot_morse_graph_viz",
    "CMGDB_MORSE_PALETTE",
    "SCIENTIFIC_MORSE_PALETTE",
    "HybridMorseComponent",
    "HybridMorsePlot",
    "hybrid_morse_components",
    "hybrid_morse_hasse",
    "PlotHybridMorseSets",
    "plot_hybrid_morse_sets",
    "save_hybrid_morse_figure",
    "ATLAS_MORSE_PLOT_SCHEMA",
    "PHYSICAL_CONLEY_FINITE_RELATION_AUDIT_SCHEMA",
    "AtlasFiniteRelationIndexAnnotations",
    "AtlasMorseBox",
    "AtlasMorseNode",
    "AtlasMorsePlotData",
    "AtlasHybridMorseComponent",
    "AtlasHybridMorsePlot",
    "extract_atlas_morse_plot_data",
    "atlas_morse_plot_data_payload",
    "save_atlas_morse_plot_data",
    "load_atlas_morse_plot_data",
    "load_atlas_finite_relation_index_annotations",
    "atlas_morse_components",
    "atlas_morse_hasse",
    "plot_atlas_hybrid_morse_sets",
    # Config
    "config",
    "configure_logging",
    "get_logger",
    # Unit-handle suspension sampling and graph diagnostics
    "BaseSuspensionSample",
    "HandleSuspensionSample",
    "SuspensionSample",
    "AugmentedCell",
    "BaseCell",
    "PhaseCell",
    "sample_suspension_trajectory",
    "simulate_suspension_endpoint",
    "locate_augmented_cells",
    "build_augmented_outer_graph",
    "crossing_completed_state",
    "simulate_crossing_completed_state",
    "collapse_phase_paths",
    "RecurrentMorseCollapseDiagnostic",
    "PhaseOnlyRecurrentComponentError",
    "diagnose_recurrent_morse_collapse",
    "assert_recurrent_morse_equivalence",
    # Reset-glued suspension cell complexes and fixed-time carriers
    "FiniteCellComplex",
    "CubicalCell",
    "CubicalGridComplex",
    "CubicalHyperplaneAttachmentAudit",
    "build_cubical_grid_complex",
    "audit_cubical_hyperplane_attachment",
    "CellularAttachmentMap",
    "CellularMapBetweenComplexes",
    "CrossComplexAcyclicCarrier",
    "DoubleMappingCylinderComplex",
    "DoubleMappingCylinderHandle",
    "CellularResetMap",
    "ResetHandle",
    "SampledSuspensionCellAdapter",
    "SparseCubicalGridComplex",
    "SuspensionBaseCell",
    "GuardPrismCell",
    "PhaseSliceCell",
    "PhaseCellRegistry",
    "SuspensionCellComplex",
    "RelativeCellPair",
    "FixedTimeCellRelation",
    "FixedTimeCarrier",
    "CellularChainMap",
    "CMGDBCellRelationPayload",
    "CMGDBRelativeHomologyPayload",
    # Actual Atlas quotient nerves and physical Conley preparation
    "AffineBoundaryEmbedding2D",
    "AtlasGoodCoverError",
    "AtlasMappingCylinderConleyPreparation",
    "AtlasMappingCylinderRegistry",
    "AtlasMappingCylinderRelativePair",
    "AtlasMappingCylinderTopCell",
    "AtlasNerveSimplex",
    "AtlasPhysicalConleyPreparation2D",
    "AtlasQuotientNerveComplex2D",
    "AtlasRectangleCell2D",
    "AtlasRelativeIndexPair2D",
    "AtlasResetGluing2D",
    "AtlasSeamSubcomplexAudit2D",
    "PhysicalConleyCertificationError",
    "QuotientIntersectionAudit2D",
    "QuotientRectangleRepresentative2D",
    "RawAtlasCellCover2D",
    "atlas_cells_from_morse_graph",
    "atlas_cells_from_phase_space",
    "prepare_atlas_physical_conley_2d",
    "prepare_atlas_mapping_cylinder_conley",
    "prepare_atlas_relation_conley_2d",
    # Full SCC reconstruction from base macro graphs
    "VirtualPhaseNode",
    "PhasePathDescriptor",
    "PhaseGadgetDescriptor",
    "PhaseDescriptor",
    "FullSCCReconstruction",
    "build_base_macro_graph",
    "reconstruct_from_base_descriptors",
    "reconstruct_full_scc_condensation",
    "FixedTimeSuspensionGridResult",
    "GridResetHandle",
    "GridSuspensionIngredients",
    "SuspensionCemeteryCell",
    "compute_fixed_time_suspension_grid",
    # Falsification and quotient-connectivity diagnostics
    "AuditedCell",
    "RelationAuditFailure",
    "EndpointProbe",
    "EndpointCoverageWitness",
    "EndpointEvaluationFailure",
    "EndpointCoverageAudit",
    "audit_endpoint_probes",
    "audit_dense_base_endpoints",
    "AugmentedQuotientIncidence",
    "CellSetConnectivity",
    "RelationConnectivityAudit",
    "audit_relation_image_connectivity",
    "audit_cell_sets_connectivity",
]
