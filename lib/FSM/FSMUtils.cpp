#include "FSM/FSMUtils.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/InliningUtils.h"
#include <mlir/IR/BlockAndValueMapping.h>

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

TransitionOp findDirectoryWinningTransition(TransitionOp racingTransition) {
  std::vector<MessageOp> allDirectoryMsgs{};
  searchForIf(
      racingTransition->getParentOfType<ModuleOp>().lookupSymbol("directory"),
      allDirectoryMsgs, [&racingTransition](MessageOp op) {
        return op.msgName() == racingTransition.sym_name();
      });
  assert(allDirectoryMsgs.size() == 1 &&
         "CONFLICT: directory sent same message in more than one state!");
  return allDirectoryMsgs.at(0)->getParentOfType<TransitionOp>();
}

LogicalResult inlineTransition(TransitionOp from, TransitionOp to,
                               PatternRewriter &rewriter) {
  // set an inline point at the end of the 'to' transition
  Block *toEntry = to.addEntryBlock();
  rewriter.setInsertionPointToStart(toEntry);
  auto inlinePoint = rewriter.create<ConstantOp>(rewriter.getUnknownLoc(),
                                                 rewriter.getI64IntegerAttr(1));

  // make sure that from has a trailing terminator
  rewriter.setInsertionPointToEnd(&from.body().front());
  auto fromTerminator = rewriter.create<BreakOp>(rewriter.getUnknownLoc());

  // create the inliner
  InlinerInterface inliner(rewriter.getContext());
  BlockAndValueMapping mapping;

  // if from has arguments map them correctly
  for(size_t arg = 0; arg < from.getNumArguments(); arg++)
    mapping.map(from.getArgument(arg), to.getArgument(arg));

  LogicalResult result = inlineRegion(
      /*inliner interface*/ inliner,
      /* src region */ &from.getRegion(),
      /* inline point */ inlinePoint,
      /* mapper */ mapping,
      /* results to replace */ {},
      /* region result types */ {});

  assert(succeeded(result) && "failed to inline transitions");

  rewriter.eraseOp(inlinePoint);
  rewriter.eraseOp(fromTerminator);
  return result;
}

LogicalResult optimizeStateTransition(StateOp startState,
                                      TransitionOp racingTransition,
                                      PatternRewriter &rewriter) {
  // set insertion point
  rewriter.setInsertionPointToEnd(startState.getBody());

  // we need to create the race handling transition
  TransitionOp handleRaceTrans = rewriter.create<TransitionOp>(
      startState.getLoc(), racingTransition.sym_name(),
      racingTransition.getType(), /*nextState*/ nullptr);

  // inline ops from the racing transition
  if (failed(inlineTransition(racingTransition, handleRaceTrans, rewriter)))
    return mlir::failure();



  return success();
}

//// set the rewriter to create a new transition at the end of the
//// current state
// rewriter.setInsertionPointToEnd(startState.getBody());
//
//// this new transition will be
//// (startState, racingTransition.action, ???)
//// currently, we do not specify a nextState
//// This we will deduce later
// TransitionOp optTrans = rewriter.create<TransitionOp>(
//     startState.getLoc(), racingEvent, racingTransaction.getType(),
//     /*nextState*/ nullptr);
//
// LogicalResult wasInlined =
//     inlineTransition(racingTransaction, optTrans, rewriter);
// assert(succeeded(wasInlined) && "failed to inline Transition");
//
/////// Now we need to find out which state to transition to /////
//
//// if we can we should transition to a state that represents
//// starting from racingTransaction.nextState and sending the message
// MessageOp lastMsgSend = getLastMessageSent(startState);
//
//// does such a state exist???
//// if so go there and stop
// std::string nextStateName =
//     racingTransaction.nextStateAttr().getLeafReference().str() + "_" +
//     startState.sym_name().str();
// if (theCache.lookupSymbol<StateOp>(nextStateName)) {
//   // update the optimised trans to go to this new state
//   optTrans.nextStateAttr(rewriter.getSymbolRefAttr(nextStateName));
//   return WalkResult::advance();
// }
//
//// next we need to find the transition in the directory that originally
//// sent us the message
// TransitionOp directoryWinningTransition =
//     findDirectoryWinningTransition(racingTransaction);
//
// assert(directoryWinningTransition.nextState().hasValue() &&
//        "cannot optimize without knowing the next state of the directory");
//
// StateOp transitionedToStateDir = theDirectory.lookupSymbol<StateOp>(
//     directoryWinningTransition.nextStateAttr());
//
//// how will our message be handled
// TransitionOp dirRaceHandle =
//     transitionedToStateDir.lookupSymbol<TransitionOp>(
//         lastMsgSend.msgName());
//
//// if this handler does not exist - skip for now // TODO - implement this
//// case
// assert(dirRaceHandle != nullptr && "unable to find correct transaction");
//
//// how does the directory respond ??
//// find if there is a top level msg -> this will always be sent
// auto dirRespMsg = dirRaceHandle.getOps<MessageOp>();
// std::vector<MessageOp> dirRespMsgs{dirRespMsg.begin(), dirRespMsg.end()};
// assert(dirRespMsgs.size() <= 1 && "sends more than one message");
// llvm::StringRef dirRespMsgName = dirRespMsgs.at(0).msgName();
//
//// we now need a new state
//// set the rewriter correctly
// rewriter.setInsertionPointToEnd(&theCache.body().front());
// StateOp nextStateOp = rewriter.create<StateOp>(
//     startState.getLoc(), nextStateName, /*isTransient*/ nullptr,
//     /*prevTransition*/
//     rewriter.getSymbolRefAttr(
//         startState.sym_name(),
//         {rewriter.getSymbolRefAttr(optTrans.sym_name())}));
// Block *nsoEntry = nextStateOp.addEntryBlock();
// rewriter.setInsertionPointToStart(nsoEntry);
//
//// optTrans will now go to this new state
// optTrans.nextStateAttr(rewriter.getSymbolRefAttr(nextStateName));
//
//// create the new transition
// TransitionOp swallowMsgTransition = rewriter.create<TransitionOp>(
//     nextStateOp.getLoc(), dirRespMsgName, racingTransaction.getType(),
//     racingTransaction.nextStateAttr());
//
// LogicalResult inlineResult = inlineTransition(
//     startState.lookupSymbol<TransitionOp>(dirRespMsgName),
//     swallowMsgTransition, rewriter);
// assert(succeeded(inlineResult) && "transaction was inlined successfully");

} // namespace utils
} // namespace fsm
} // namespace mlir