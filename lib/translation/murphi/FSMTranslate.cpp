#include "FSM/FSMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Translation.h"
#include "translation/murphi/Translate.h"

namespace mlir {
void registerFsmToMurphiTranslation() {
  mlir::TranslateFromMLIRRegistration registration(
      "fsm-to-murphi",
      [](mlir::ModuleOp, mlir::raw_ostream &) -> LogicalResult {
        // TODO - implement fsm translation
        return mlir::success();
      },
      [](mlir::DialectRegistry &registry) {
        registry.insert<mlir::fsm::FSMDialect>();
        registry.insert<StandardOpsDialect>();
      });
}

} // namespace mlir