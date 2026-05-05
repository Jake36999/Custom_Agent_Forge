"""Quick smoke test for all new/modified models and modules."""
from src.core.models import Constraint, Trajectory, TrajectoryStep, CognitiveNode, ReasoningEdge, AletheiaSkill, SemanticReasoningNode
from src.pipeline.telemetry import TelemetryCollector
from src.pipeline.contracts import validate_contract, MODE_CONTRACTS
from src.pipeline.reroll import RerollEngine
from src.pipeline.identity_manager import IdentityManager

# Test Constraint
c = Constraint(type='structural', description='Missing docstring', severity='warning', tags=['docstring'])
print(f"Constraint: {c.type} / {c.severity} / valid={c.valid}")

# Test Trajectory
t = Trajectory(
    steps=[TrajectoryStep(attempt_code='x=1', error='NameError', diagnosis='Undefined', fix='x=0')],
    confusion_matrix=[[1.0, 0.2]],
    response_vector=[0.8],
)
print(f"Trajectory: {len(t.steps)} steps, matrix={t.confusion_matrix}")

# Test CognitiveNode (minimal)
skill = AletheiaSkill(
    node_id='test_1', name='test', file='test.py', code_snippet='pass',
    imports=[], operator_type='function',
    teaching_layer={'skill_identity': {}, 'method_metadata': {}},
    epistemic={'state': 'CREATED', 'c_node': 0.5},
)
cn = CognitiveNode(cognitive_id='test_1', skill=skill)
print(f"CognitiveNode: {cn.cognitive_id}, mode={cn.mode}, c_final={cn.c_final}")

# Test ReasoningEdge
edge = ReasoningEdge(source_id='a', target_id='b', edge_type='dependency')
print(f"ReasoningEdge: {edge.source_id} -> {edge.target_id}")

# Test TelemetryCollector
tc = TelemetryCollector()
tc.record('slr', 'breach_detected', node_id='n1', payload={'slr': 0.72, 'conviction': 0.28})
tc.record('mode', 'execution', node_id='n1', payload={'mode': 'advocate'})
tc.record('sie', 'computed', node_id='n1', payload={'s_sie': 0.85})
tc.record('identity', 'drift_update', node_id='n1', payload={'drift_increment': 0.072})
snap = tc.snapshot()
print(f"Telemetry: {snap['total_events']} events, SLR mean={snap['slr_distribution']['mean']}, mode_balance={snap['mode_balance']}")

print("\nALL IMPORTS AND INSTANTIATIONS OK")
