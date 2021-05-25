#include "translation/murphi/TranslatePCCToMurphi.h"
#include "PCC/PCCDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Translation.h"
//#include "models/Expr.h"

using namespace mlir::pcc;

// private namespace for the implementation
namespace {

class MurphiTranslateImpl {
public:
  MurphiTranslateImpl(mlir::ModuleOp op, mlir::raw_ostream &output)
      : theModule{op}, output{output} {}
  mlir::LogicalResult translate() {
    //    murphi::IntExpr test(21);
    //    test.getAsString();
    return mlir::success();
  }

private:
  mlir::ModuleOp theModule;
  mlir::raw_ostream &output;
};

} // namespace

namespace mlir {
void registerToMurphiTranslation() {
  mlir::TranslateFromMLIRRegistration registration(
      "pcc-to-murphi",
      [](mlir::ModuleOp op, mlir::raw_ostream &output) {
        MurphiTranslateImpl murphiTranslate(op, output);
        return murphiTranslate.translate();
      },
      [](mlir::DialectRegistry &registry) {
        registry.insert<PCCDialect>();
        registry.insert<mlir::StandardOpsDialect>();
      });
}

} // namespace mlir