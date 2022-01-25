#pragma once
#include "FSM/FSMOps.h"
#include <algorithm>
#include <mlir/IR/PatternMatch.h>

namespace mlir {
namespace fsm {
namespace utils {
/// This helper function searches through all nested regions
/// and pushes to the vector all matched operations
/// is is able to be cast to the template type
template <class OpType, class CallableT>
void searchForIf(Operation *op, std::vector<OpType> &matchingOps,
                 CallableT filter) {
  if (OpType castedOp = mlir::dyn_cast<OpType>(op))
    if (filter(castedOp))
      matchingOps.push_back(castedOp);

  // iterate through all regions and call recursively
  for (auto &region : op->getRegions()) {
    for (auto &regionOps : region.getOps()) {
      searchForIf(&regionOps, matchingOps, filter);
    }
  }
}

template <class OpType>
void searchFor(Operation *op, std::vector<OpType> &matchingOps) {
  return searchForIf<OpType, bool(OpType)>(op, matchingOps,
                                           [](OpType) { return true; });
}

/// Returns true if the state op represents a transient state
bool isTransientState(StateOp stateOp);

/// checks if the current action is a CPU event
bool isCpuEvent(llvm::StringRef val);
bool isCpuEvent(TransitionOp op);

/// returns the previous linked transition (op must have linked previous state)
TransitionOp getPreviousTransition(StateOp op);

/// Looks through linked previous transitions to find which message
/// was sent to the directory previously
MessageOp getLastMessageSent(StateOp op);

/// Recursively looks back through linked previous transition
/// to find the original start state
StateOp getStableStartState(StateOp stateOp);

/// returns the transition that 'won' the race, before our message is received
TransitionOp findDirectoryWinningTransition(TransitionOp racingTransition);

LogicalResult inlineTransition(TransitionOp from, TransitionOp to, PatternRewriter &rewriter);

LogicalResult optimizeStateTransition(StateOp startState, TransitionOp racingTransition, PatternRewriter &rewriter);

} // namespace utils
} // namespace fsm
} // namespace mlir