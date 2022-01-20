#include "FSM/FSMUtils.h"

namespace mlir {
namespace fsm {
namespace utils {
bool isTransientState(StateOp stateOp) {
  // a state is a transient state if it satisfies one of the following rules

  // 1. state is labeled with isTransient and is set to true
  // 2. has a previous transition attr
  // 3. if state is not a single letter -> transient state

  bool rule1 =
      stateOp.isTransient().hasValue() && stateOp.isTransient().getValue();
  bool rule2 = stateOp.prevTransition().hasValue();
  bool rule3 = stateOp.sym_name().size() != 1;
  // branch optimization
  return static_cast<bool>(rule1 + rule2 + rule3);
}

bool isCpuEvent(llvm::StringRef val) {
  return val == "load" || val == "store" || val == "evict";
}

bool isCpuEvent(TransitionOp op) { return isCpuEvent(op.sym_name()); }

TransitionOp getPreviousTransition(StateOp op) {
  assert(op.prevTransition().hasValue() &&
         "state op not labeled with previous transition");
  MachineOp parentMachine = op->getParentOfType<MachineOp>();
  SymbolRefAttr refAttr = op.prevTransitionAttr();
  return parentMachine.lookupSymbol<TransitionOp>(refAttr);
}

MessageOp getLastMessageSent(StateOp op) {
  // currently, we only look back 1 level
  // maybe this can be changed to look arbitrarily far but that is to be seen
  TransitionOp previousTransition = getPreviousTransition(op);
  Block &transitionsBlock = previousTransition.front();
  // return the first message found
  // maybe an error???
  for (auto msgOp : transitionsBlock.getOps<MessageOp>()) {
    return msgOp;
  }
  assert(false && "linked previous state sent no messages");
}

StateOp getStableStartState(StateOp stateOp) {
  MachineOp parentMach = stateOp->getParentOfType<MachineOp>();
  // if a stable state -> return self
  if (!isTransientState(stateOp))
    return stateOp;

  assert(stateOp.prevTransition().hasValue() && "no linked previous state!");
  TransitionOp prevTransition =
      parentMach.lookupSymbol<TransitionOp>(stateOp.prevTransitionAttr());
  assert(prevTransition != nullptr &&
         "could not lookup the correct link from prevTransition");
  return getStableStartState(prevTransition->getParentOfType<StateOp>());
}
} // namespace utils
} // namespace fsm
} // namespace mlir