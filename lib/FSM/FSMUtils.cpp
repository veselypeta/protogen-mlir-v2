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

  // look back at most 2 states - for now
  // FIXME -> This needs a more general solution
  if (auto prePreviousTransition = getPreviousTransition(
          previousTransition->getParentOfType<StateOp>())) {
    for (auto msgOp : prePreviousTransition.front().getOps<MessageOp>()) {
      return msgOp;
    }
  }

  assert(false && "linked previous state sent no messages");
}

llvm::StringRef getLastActionPerformed(StateOp op) {
  // FIXME - looks back at most two states
  TransitionOp previousTransition = getPreviousTransition(op);
  if (isCpuEvent(previousTransition.sym_name()))
    return previousTransition.sym_name();

  previousTransition =
      getPreviousTransition(previousTransition->getParentOfType<StateOp>());
  if (isCpuEvent(previousTransition.sym_name()))
    return previousTransition.sym_name();
  assert(0 && "could not find a cpu event linking previous transitions");
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
  auto inlinePoint = rewriter.create<NOPOp>(rewriter.getUnknownLoc());

  // make sure that from has a trailing terminator
  rewriter.setInsertionPointToEnd(&from.body().front());
  auto fromTerminator = rewriter.create<BreakOp>(rewriter.getUnknownLoc());

  // create the inliner
  InlinerInterface inliner(rewriter.getContext());
  BlockAndValueMapping mapping;

  // if from has arguments map them correctly
  for (size_t arg = 0; arg < from.getNumArguments(); arg++)
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

std::string getNextStateName(llvm::StringRef destinationState,
                             llvm::StringRef actionPerformed) {
  std::string newState = destinationState.str() + "_" + actionPerformed.str();
  return newState;
}

TransitionOp getDirectoryTransitionHavingSent(llvm::StringRef msgName,
                                              MachineOp theDirectory) {
  std::vector<MessageOp> allSentMessages;
  searchForIf(theDirectory, allSentMessages, [msgName](MessageOp test_msg) {
    return test_msg.msgName() == msgName;
  });
  assert(allSentMessages.size() == 1 &&
         "directory message sending should map to a unique value");
  TransitionOp executedTransition =
      allSentMessages.at(0)->getParentOfType<TransitionOp>();
  return executedTransition;
}
StateOp getDirectoryDestinationStateHavingSent(llvm::StringRef msgName,
                                               MachineOp theDirectory) {
  auto executedTransition =
      getDirectoryTransitionHavingSent(msgName, theDirectory);
  assert(executedTransition.nextState().hasValue() &&
         "undecidable unless directory has known end state");

  auto dst_state =
      theDirectory.lookupSymbol<StateOp>(executedTransition.nextStateAttr());
  if (!isTransientState(dst_state))
    return dst_state;

  // TODO - this is a bit of a hack
  // else find what state it is assigned to
  std::vector<UpdateOp> stateUpdates;
  searchForIf(dst_state, stateUpdates, [](UpdateOp updateOp) {
    return updateOp.variable().getType().isa<StateType>();
  });
  assert(stateUpdates.size() == 1 &&
         "more than one state update occurs in the transition");
  auto theConstOp = stateUpdates.at(0).value().getDefiningOp<ConstOp>();
  assert(theConstOp != nullptr && "cast to ConstOp failed!");
  auto theState = theDirectory.lookupSymbol<StateOp>(
      theConstOp.value().cast<StringAttr>().getValue());
  return theState;
}

llvm::Optional<StateOp> getDestinationState(StateOp startState,
                                            TransitionOp racingTransition) {
  auto theCache = startState->getParentOfType<MachineOp>();
  auto prev_msg_sent = getLastMessageSent(startState);
  auto prev_action_performed = getLastActionPerformed(startState);
  auto racing_trans_end_state =
      racingTransition.nextStateAttr().getLeafReference();
  auto end_state_by_msg =
      racing_trans_end_state.str() + "_" + prev_msg_sent.msgName().str();
  auto end_state_by_action =
      racing_trans_end_state.str() + "_" + prev_action_performed.str();
  if (auto by_msg = theCache.lookupSymbol<StateOp>(end_state_by_msg)) {
    return {by_msg};
  } else if (auto by_action =
                 theCache.lookupSymbol<StateOp>(end_state_by_action)) {
    return {by_action};
  } else {
    return {};
  }
}

MessageOp getDirectoryResponseMessage(TransitionOp op) {

  if (!op.getOps<fsm::CallOp>().empty()) {
    auto callOps = op.getOps<fsm::CallOp>();
    auto callOp = *callOps.begin();
    auto calledTrans =
        op->getParentOfType<StateOp>().lookupSymbol<TransitionOp>(
            callOp.theTransition());
    assert(calledTrans != nullptr &&
           "could not find called transition in the current state");
    op = calledTrans;
  }
  std::vector<MessageOp> dir_resps;
  searchFor(op, dir_resps);
  assert(dir_resps.size() == 1 && "accept only a single response from the dir");
  return dir_resps.at(0);
}

LogicalResult handleDestinationStateExists(StateOp dstState, StateOp startState,
                                           TransitionOp racingTransition,
                                           PatternRewriter &rewriter) {
  // create the transition to handle the racing message
  rewriter.setInsertionPointToEnd(&startState.transitions().front());
  TransitionOp handleRaceTrans = rewriter.create<TransitionOp>(
      startState.getLoc(), racingTransition.sym_name(),
      racingTransition.getType(),
      /* nextState */ rewriter.getSymbolRefAttr(dstState));

  // inline
  if (failed(inlineTransition(racingTransition, handleRaceTrans, rewriter)))
    return failure();

  // check the directory for reinterpretations
  auto prev_msg_sent = getLastMessageSent(startState);

  // what state will the directory be when it's received?
  MachineOp theDirectory =
      startState->getParentOfType<ModuleOp>().lookupSymbol<MachineOp>(
          "directory");
  StateOp dir_state_when_received = getDirectoryDestinationStateHavingSent(
      racingTransition.sym_name(), theDirectory);

  emitRemark(handleRaceTrans.getLoc(),
             "Optimized (" + startState.sym_name() + ", " +
                 handleRaceTrans.sym_name() + ") -> " +
                 handleRaceTrans.nextStateAttr().getLeafReference());
  // can the directory handle this message
  if (dir_state_when_received.lookupSymbol<TransitionOp>(
          prev_msg_sent.msgName()) != nullptr)
    return success();

  // the directory should re-interpret this message
  auto dir_expected_msg = getLastMessageSent(dstState);
  auto dirHandler = dir_state_when_received.lookupSymbol<TransitionOp>(
      dir_expected_msg.msgName());
  assert(dirHandler != nullptr && "could not resolve the directory handler");

  rewriter.setInsertionPointAfter(dirHandler);
  auto reinterpretedDirMsg = rewriter.create<TransitionOp>(
      dirHandler.getLoc(), prev_msg_sent.msgName(), dirHandler.getType(),
      dirHandler.nextStateAttr());
  // call the correct handler
  auto RIDMBlock = reinterpretedDirMsg.addEntryBlock();
  rewriter.setInsertionPointToStart(RIDMBlock);
  rewriter.create<fsm::CallOp>(reinterpretedDirMsg.getLoc(),
                               rewriter.getSymbolRefAttr(dirHandler.sym_name()),
                               RIDMBlock->getArgument(0));
  emitRemark(reinterpretedDirMsg.getLoc(),
             "Added Reinterpretation at directory (" +
                 dir_state_when_received.sym_name() + ", " +
                 reinterpretedDirMsg.sym_name() + ")");
  return success();
}

LogicalResult optimizeStateTransitionV2(StateOp startState,
                                        TransitionOp racingTransition,
                                        PatternRewriter &rewriter) {
  //  auto theCache = startState->getParentOfType<MachineOp>();
  auto theDirectory =
      startState->getParentOfType<ModuleOp>().lookupSymbol<MachineOp>(
          "directory");

  // if a dst state exists already
  auto dst_state = getDestinationState(startState, racingTransition);
  if (dst_state.hasValue()) {
    return handleDestinationStateExists(dst_state.getValue(), startState,
                                        racingTransition, rewriter);
  }

  // if the dst_state does not exist, i.e. in the M_evict -> Fwd_GetM case
  // i.e. would be looking for I_evict or I_PutM

  // we need a new state to transition to
  // we can compute what the next state will be called
  auto prev_msg_sent = getLastMessageSent(startState);
  auto next_state_name =
      getNextStateName(racingTransition.nextState()->getLeafReference(),
                       prev_msg_sent.msgName());

  // we need to create the race handling transition
  rewriter.setInsertionPointToEnd(&startState.transitions().front());
  TransitionOp handleRaceTrans = rewriter.create<TransitionOp>(
      startState.getLoc(), racingTransition.sym_name(),
      racingTransition.getType(),
      /*nextState*/ rewriter.getSymbolRefAttr(next_state_name));

  // inline ops from the racing transition
  if (failed(inlineTransition(racingTransition, handleRaceTrans, rewriter)))
    return mlir::failure();

  emitRemark(handleRaceTrans.getLoc(),
             "Optimized: (" + startState.sym_name() + ", " +
                 racingTransition.sym_name() + ") -> " +
                 handleRaceTrans.nextStateAttr().getLeafReference());

  // create the needed new state
  rewriter.setInsertionPointAfter(startState);
  StateOp nextState = rewriter.create<StateOp>(
      startState.getLoc(), /*sym_name*/ next_state_name,
      /*isTransient*/ nullptr,
      /*prev_transition*/
      rewriter.getSymbolRefAttr(handleRaceTrans.sym_name()));

  // what message did we last send?
  MessageOp last_msg_sent = getLastMessageSent(startState);

  // what state will the directory be when it's received?
  StateOp dir_state_when_received = getDirectoryDestinationStateHavingSent(
      racingTransition.sym_name(), theDirectory);
  assert(dir_state_when_received != nullptr && "couldn't find dir state");

  // can the directory handle the message we sent in this state?
  TransitionOp dir_handling_trans =
      dir_state_when_received.lookupSymbol<TransitionOp>(
          last_msg_sent.msgName());
  if (dir_handling_trans == nullptr) {
    // the directory does not have a handler for this message in the current
    // state
    assert(0 && "currently we do not handle this case!");
  }

  // what message will the directory respond with?
  auto dir_resp = getDirectoryResponseMessage(dir_handling_trans);
  // in the new state we now create the new transition to accept this message
  auto nsEntry = nextState.addEntryBlock();
  rewriter.setInsertionPointToStart(nsEntry);
  TransitionOp handle_dir_resp = rewriter.create<TransitionOp>(
      racingTransition->getLoc(),
      /*action*/ dir_resp.msgName(),
      /*type*/
      rewriter.getFunctionType({MsgType::get(rewriter.getContext())}, {}),
      /*nextState*/ racingTransition.nextStateAttr());

  TransitionOp dir_resp_src_ops =
      startState.lookupSymbol<TransitionOp>(dir_resp.msgName());

  // inline the ops
  return inlineTransition(dir_resp_src_ops, handle_dir_resp, rewriter);
}

LogicalResult optimizeStateTransition(StateOp startState,
                                      TransitionOp racingTransition,
                                      PatternRewriter &rewriter) {
  MachineOp theCache = startState->getParentOfType<MachineOp>();
  MachineOp theDirectory =
      startState->getParentOfType<ModuleOp>().lookupSymbol<MachineOp>(
          "directory");

  // set insertion point
  rewriter.setInsertionPointToEnd(startState.getBody());

  // we can compute what the next state will be called
  auto prev_msg_sent = getLastMessageSent(startState);
  auto next_state_name =
      getNextStateName(racingTransition.nextState()->getLeafReference(),
                       prev_msg_sent.msgName());

  // we need to create the race handling transition
  TransitionOp handleRaceTrans = rewriter.create<TransitionOp>(
      startState.getLoc(), racingTransition.sym_name(),
      racingTransition.getType(),
      /*nextState*/ rewriter.getSymbolRefAttr(next_state_name));

  // inline ops from the racing transition
  if (failed(inlineTransition(racingTransition, handleRaceTrans, rewriter)))
    return mlir::failure();

  // if the new state already exists we are done
  // else we need to deduce how the directory will respond to us
  if (theCache.lookupSymbol<StateOp>(next_state_name))
    return success();

  // create the needed new state
  rewriter.setInsertionPointAfter(startState);
  StateOp nextState = rewriter.create<StateOp>(
      startState.getLoc(), /*sym_name*/ next_state_name,
      /*isTransient*/ nullptr,
      /*prev_transition*/
      rewriter.getSymbolRefAttr(handleRaceTrans.sym_name()));

  // what message did we last send?
  MessageOp last_msg_sent = getLastMessageSent(startState);

  // what state will the directory be when it's received?
  StateOp dir_state_when_received = getDirectoryDestinationStateHavingSent(
      racingTransition.sym_name(), theDirectory);

  // can the directory handle the message we sent in this state?
  TransitionOp dir_handling_trans =
      dir_state_when_received.lookupSymbol<TransitionOp>(
          last_msg_sent.msgName());
  if (dir_handling_trans == nullptr)
    assert(0 && "currently we do not handle this case!");

  // what message will the directory respond with?
  std::vector<MessageOp> dir_resps;
  searchFor(dir_handling_trans, dir_resps);
  assert(dir_resps.size() == 1 && "accept only a single response from the dir");

  // in the new state we now create the new transition to accept this message
  auto nsEntry = nextState.addEntryBlock();
  rewriter.setInsertionPointToStart(nsEntry);
  TransitionOp handle_dir_resp = rewriter.create<TransitionOp>(
      racingTransition->getLoc(),
      /*action*/ dir_resps.at(0).msgName(),
      /*type*/
      rewriter.getFunctionType({MsgType::get(rewriter.getContext())}, {}),
      /*nextState*/ racingTransition.nextStateAttr());

  TransitionOp dir_resp_src_ops =
      startState.lookupSymbol<TransitionOp>(dir_resps.at(0).msgName());

  // inline the ops
  return inlineTransition(dir_resp_src_ops, handle_dir_resp, rewriter);
}

llvm::Optional<StateOp> getNextStateIfPossible(StateOp s) {
  auto iTransitions = s.getOps<TransitionOp>();
  // we use adjacent find -> return false wherever they do match
  auto find = std::adjacent_find(
      std::begin(iTransitions), std::end(iTransitions),
      [](TransitionOp lhs, TransitionOp rhs) {
        if (lhs.nextState().hasValue() && rhs.nextState().hasValue())
          return lhs.nextStateAttr().getLeafReference() !=
                 lhs.nextStateAttr().getLeafReference();
        return true;
      });
  // if it got to the end -> all are equal;
  bool allEqual = find == std::end(iTransitions);
  if (allEqual) {
    MachineOp theCache = s->getParentOfType<MachineOp>();

    return {theCache.lookupSymbol<StateOp>(
        (*std::begin(iTransitions)).nextStateAttr())};
  }
  return {};
}

llvm::Optional<StateOp> getNonStallingEndStateIfPossible(StateOp tState) {
  StateOp nextState;
  do {
    auto possibleNextState = getNextStateIfPossible(tState);
    if (possibleNextState.hasValue())
      nextState = possibleNextState.getValue();
    else
      return {};
  } while (isTransientState(nextState));
  return {nextState};
}
} // namespace utils
} // namespace fsm
} // namespace mlir