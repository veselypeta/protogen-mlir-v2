#include "PCC/Passes/Passes.h"
#include "PassDetail.h"
using namespace mlir;
namespace {

struct TransientStatesPass
    : public pcc::TransientStatesBase<TransientStatesPass> {
  void runOnOperation() override;
};

void TransientStatesPass::runOnOperation() {}

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> pcc::createTransientStatesPass() {
  return std::make_unique<TransientStatesPass>();
}