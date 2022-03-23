//
// Created by petr on 23/03/2022.
//
#include "FSM/FSMUtils.h"
#include "FSM/Passes/Passes.h"
#include "PassDetail.h"
#include "ProtoGenRewriter.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Pass/PassManager.h"
#include <mlir/IR/BlockAndValueMapping.h>

using namespace mlir;
using namespace mlir::fsm;
namespace {
class RemoveAwaitPass : public RemoveAwaitPassBase<RemoveAwaitPass> {
public:
  void runOnOperation() override;
};

void RemoveAwaitPass::runOnOperation() {
  ModuleOp theModule = OperationPass<ModuleOp>::getOperation();

  auto result = theModule.walk([&](AwaitOp awaitOp) -> WalkResult {
    auto parentOp = awaitOp->getParentOp();
    ProtoGenRewriter rewriter(&getContext());

    if (!llvm::isa<TransitionOp>(parentOp))
      return WalkResult::advance();
    auto parentTrans = dyn_cast<TransitionOp>(parentOp);
    auto parentState = parentTrans->getParentOfType<StateOp>();
    rewriter.setInsertionPointAfter(parentState);

    auto parentStateName = parentState.sym_name();
    auto transStateName = parentStateName + "_" + parentTrans.sym_name();
    auto newTransState = rewriter.create<StateOp>(awaitOp.getLoc(), transStateName.str());
    parentTrans.nextStateAttr(rewriter.getSymbolRefAttr(newTransState));

    rewriter.setInsertionPointToStart(newTransState.addEntryBlock());

    auto whenResult = awaitOp.walk([&](WhenOp whenOp)->WalkResult{
      auto tFnType = rewriter.getFunctionType({MsgType::get(&getContext())}, {});
      auto nTransition = rewriter.create<TransitionOp>(whenOp->getLoc(),
                                                       whenOp.sym_name(), tFnType, nullptr);
      assert(succeeded(utils::inlineWhenIntoTrans(whenOp, nTransition, rewriter)));
      rewriter.eraseOp(whenOp);
      return WalkResult::advance();
    });
    if (whenResult.wasInterrupted())
      return WalkResult::interrupt();

    rewriter.eraseOp(awaitOp);
    return WalkResult::advance();
  });

  if (result.wasInterrupted())
    return signalPassFailure();
}

} // namespace

namespace mlir {
namespace fsm {
std::unique_ptr<OperationPass<ModuleOp>> createRemoveAwaitPass() {
  return std::make_unique<RemoveAwaitPass>();
}
} // namespace fsm
} // namespace mlir