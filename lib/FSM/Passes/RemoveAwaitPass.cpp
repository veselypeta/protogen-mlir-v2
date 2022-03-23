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
#include "iostream"


using namespace mlir;
using namespace mlir::fsm;
namespace {
class RemoveAwaitPass : public RemoveAwaitPassBase<RemoveAwaitPass> {
public:
  void runOnOperation() override;
};

void RemoveAwaitPass::runOnOperation() {
  ModuleOp theModule = OperationPass<ModuleOp>::getOperation();
  bool change = false;

  auto handler = [&](AwaitOp awaitOp) -> WalkResult {
    auto parentOp = awaitOp->getParentOp();
    ProtoGenRewriter rewriter(&getContext());

    // only consider awaits which have parent Transition
    if (!llvm::isa<TransitionOp>(parentOp))
      return WalkResult::advance();

    auto parentTransOp = dyn_cast<TransitionOp>(parentOp);
    auto parentStateOp = parentTransOp->getParentOfType<StateOp>();
    rewriter.setInsertionPointAfter(parentStateOp);

    auto parentStateName = parentStateOp.sym_name().str();
    auto transStateName = parentStateName + "_" + parentTransOp.sym_name().str();
    auto newTransState = rewriter.create<StateOp>(
        awaitOp.getLoc(), transStateName, rewriter.getBoolAttr(true),
        rewriter.getSymbolRefAttr(
            parentStateOp.sym_name(),
            {rewriter.getSymbolRefAttr(parentTransOp.sym_name())}));
    parentTransOp.nextStateAttr(rewriter.getSymbolRefAttr(newTransState));

    rewriter.setInsertionPointToStart(newTransState.addEntryBlock());

    for(auto whenOp : awaitOp.body().getOps<WhenOp>()){
      auto tFnType =
          rewriter.getFunctionType({MsgType::get(&getContext())}, {});
      auto nTransition = rewriter.create<TransitionOp>(
          whenOp->getLoc(), whenOp.sym_name(), tFnType, nullptr);
      assert(
          succeeded(utils::inlineWhenIntoTrans(whenOp, nTransition, rewriter)));
      rewriter.setInsertionPointAfter(nTransition);
    }
    change = true;
    rewriter.eraseOp(awaitOp);
    return WalkResult::advance();
  };

  do{
    change = false;
    auto result = theModule.walk(handler);
    if (result.wasInterrupted())
      return signalPassFailure();
  } while(change);


}

} // namespace

namespace mlir {
namespace fsm {
std::unique_ptr<OperationPass<ModuleOp>> createRemoveAwaitPass() {
  return std::make_unique<RemoveAwaitPass>();
}
} // namespace fsm
} // namespace mlir