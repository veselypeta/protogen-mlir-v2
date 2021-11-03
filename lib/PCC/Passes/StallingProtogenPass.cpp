#include "PassDetail.h"

using namespace mlir;
using namespace mlir::pcc;
namespace {

class StallingProtogenOptimizationPass
    : public StallingProtogenOptimizationPassBase<
          StallingProtogenOptimizationPass> {
public:
  void runOnOperation() override;

};

void StallingProtogenOptimizationPass::runOnOperation() {

  ModuleOp theModule = OperationPass<ModuleOp>::getOperation();

  auto result = theModule.walk([&](ProcessOp processOp){
    return WalkResult::advance();
  });

  if(result.wasInterrupted())
    return signalPassFailure();
}


} // namespace

namespace mlir {
namespace pcc {

std::unique_ptr<OperationPass<ModuleOp>>
createStallingProtogenOptimizationPass() {
  return std::make_unique<StallingProtogenOptimizationPass>();
}

} // namespace pcc
} // namespace mlir