//
// Created by petr on 10/01/2022.
//
#include "FSM/FSMUtils.h"
#include "FSM/Passes/Passes.h"
#include "PassDetail.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/InliningUtils.h"
#include <mlir/IR/BlockAndValueMapping.h>

using namespace mlir;
using namespace mlir::fsm;
using namespace mlir::fsm::utils;

// Private Namespace for Implementation
namespace {

// We need a custom rewriter to be able to construct it
class ProtoGenRewriter : public PatternRewriter {
public:
  explicit ProtoGenRewriter(MLIRContext *ctx) : PatternRewriter{ctx} {
    // TODO - overwrite necessary methods here
  }
};

// Create the Optimization Pass
class StallingOptimizationPass
    : public StallingOptimizationPassBase<StallingOptimizationPass> {
public:
  void runOnOperation() override;
};

// Implement the runOnOperation function
void StallingOptimizationPass::runOnOperation() {
  /// grab the module/cache/directory
  ModuleOp theModule = OperationPass<ModuleOp>::getOperation();

  ProtoGenRewriter rewriter(&getContext());

  // walk each transient state
  auto result = theModule.walk([&](StateOp startState) {
    // Skip non-transient states
    if (!isTransientState(startState))
      return WalkResult::advance();

    // Now we need to find the stable start state
    StateOp logicalStartState = getStableStartState(startState);

    auto racingResult =
        logicalStartState.walk([&](TransitionOp racingTransaction) {
          // skip if we can already handle this message in the current state
          // or is a cpu event
          llvm::StringRef racingEvent = racingTransaction.sym_name();
          if (startState.lookupSymbol<TransitionOp>(racingEvent) ||
              isCpuEvent(racingEvent)) {
            emitRemark(startState.getLoc(), "Skipping: (" +
                                                startState.sym_name() + ", " +
                                                racingEvent + ")");
            return WalkResult::advance();
          }

          emitRemark(startState.getLoc(), "Optimizing: (" +
                                              startState.sym_name() + ", " +
                                              racingEvent);

          // set the rewriter to create a new transition at the end of the
          // current state
          rewriter.setInsertionPointToEnd(startState.getBody());

          // this new transition will be
          // (startState, racingTransition.action, ???)
          // currently, we do not specify a nextState
          // This we will deduce later
          TransitionOp optTrans = rewriter.create<TransitionOp>(
              startState.getLoc(), racingEvent, racingTransaction.getType(),
              /*nextState*/ nullptr);
          Block *optEntry = optTrans.addEntryBlock();
          rewriter.setInsertionPointToStart(optEntry);
          // create a dummy op which is later replaced
          auto inlinerOp = rewriter.create<ConstantOp>(
              optTrans.getLoc(), rewriter.getI64IntegerAttr(1));

          // pre-emptively add a break operation to the end of the racing
          // transaction we are wanting to inline
          rewriter.setInsertionPointToEnd(&racingTransaction.body().front());
          BreakOp preemptiveBreak = rewriter.create<BreakOp>(rewriter.getUnknownLoc());

          // we have to inline the contents of the racing transaction into this
          // new transaction i.e. copy the actions
          InlinerInterface inliner(&getContext());
          BlockAndValueMapping mapping;
          mapping.map(racingTransaction.getArgument(0),
                      optTrans.getArgument(0));

          LogicalResult wasInlined = inlineRegion(
              /*inliner interface*/ inliner,
              /* src region */ &racingTransaction.getRegion(),
              /* inline point */ inlinerOp,
              /* mapper */ mapping,
              /* results to replace */ {},
              /* region result types */ {});

          assert(succeeded(wasInlined) && "failed to inline Transition");

          // remove the unused inline const op and break
          rewriter.eraseOp(inlinerOp);
          rewriter.eraseOp(preemptiveBreak);

          ///// Now we need to find out which state to transition to /////

          // What message have we originally sent to the directory???
          /*MessageOp lastMsgSent = */ getLastMessageSent(startState);

          // Which message did the directory respond to??
          // by sending us a message we can deduce in what state the directory
          // will be when it receives our message

          // What state is the directory in when it receives this message???

          // advance to next transition
          return WalkResult::advance();
        });

    if (racingResult.wasInterrupted())
      signalPassFailure();
    return WalkResult::advance();
  });

  if (result.wasInterrupted())
    return signalPassFailure();
}

} // namespace

namespace mlir {
namespace fsm {

std::unique_ptr<OperationPass<ModuleOp>> createStallingOptimizationPass() {
  return std::make_unique<StallingOptimizationPass>();
}

} // namespace fsm
} // namespace mlir