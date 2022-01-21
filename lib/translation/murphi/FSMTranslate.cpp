#include "FSM/FSMDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Translation.h"
#include "translation/murphi/Translate.h"
#include "translation/murphi/codegen/MurphiCodeGen.h"
#include "translation/murphi/codegen/FSMDialectInterpreter.h"

namespace mlir {
void registerFsmToMurphiTranslation() {
  using namespace murphi;
  mlir::TranslateFromMLIRRegistration registration(
      "fsm-to-murphi",
      [](mlir::ModuleOp moduleOp, llvm::raw_ostream &output) -> LogicalResult {
        FSMDialectInterpreter interpreter(moduleOp);
        MurphiAssembler<FSMDialectInterpreter> assembler{interpreter};
        return renderMurphi(assembler.assemble(), output);
      },
      [](mlir::DialectRegistry &registry) {
        registry.insert<mlir::fsm::FSMDialect>();
        registry.insert<StandardOpsDialect>();
      });
}

} // namespace mlir