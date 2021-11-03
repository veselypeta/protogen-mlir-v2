#include "PassDetail.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace mlir::pcc;

namespace {

class EndStatesRewriter : public PatternRewriter {
public:
  explicit EndStatesRewriter(MLIRContext *ctx) : PatternRewriter(ctx) {}
  // TODO - override any necessary methods here
};

struct EndStatesPass : EndStatesPassBase<EndStatesPass> {
  void runOnOperation() override;
};

void EndStatesPass::runOnOperation() {
  ModuleOp moduleOp = OperationPass<ModuleOp>::getOperation();
  EndStatesRewriter rewriter(&getContext());

  WalkResult result = moduleOp.walk([&](ProcessOp processOp) {
    // must contain end_state attr
    if (!processOp->hasAttr("end_state"))
      return WalkResult::advance();

    // Does not contain a transaction op
    auto nestedOpsIt = processOp.getBody().getOps();
    if (std::find_if(std::begin(nestedOpsIt), std::end(nestedOpsIt),
                     [](Operation &op) {
                       return dyn_cast<TransactionOp>(op) != nullptr;
                     }) != nestedOpsIt.end())
      return WalkResult::advance();

    llvm::StringRef endState =
        processOp->getAttr("end_state").cast<StringAttr>().getValue();
    Block &entryBlock = processOp.getBody().front();

    rewriter.setInsertionPointToEnd(&entryBlock);
    Value stateValue = rewriter.create<InlineConstOp>(
        processOp.getLoc(),
        StateAttr::get(StateType::get(&getContext(), endState.str())));
    rewriter.create<StateUpdateOp>(processOp->getLoc(),
                                   processOp.getArgument(0), stateValue,
                                   rewriter.getStringAttr("State"));

    return WalkResult::advance();
  });

  if (result.wasInterrupted())
    signalPassFailure();
}

} // namespace

namespace mlir {
namespace pcc {
std::unique_ptr<OperationPass<ModuleOp>> createEndStatesPass() {
  return std::make_unique<EndStatesPass>();
}
} // namespace pcc
} // namespace mlir
