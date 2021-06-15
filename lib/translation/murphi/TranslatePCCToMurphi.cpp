#include "translation/murphi/TranslatePCCToMurphi.h"
#include "PCC/PCCDialect.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Translation.h"
#include "models/Expr.h"

using namespace mlir::pcc;

// private namespace for the implementation
namespace {

class MurphiTranslateImpl {
public:
  MurphiTranslateImpl(mlir::ModuleOp op, mlir::raw_ostream &output)
      : theModule{op}, output{output} {}
  mlir::LogicalResult translate() {
    generatePreamble();

    // Translate Top Level Constructs (Constants ... )


    return mlir::success();
  }

  void generatePreamble(){
    repeatChar('-', 20);
    output << '\n';
    repeatChar('-', 2);
    output << " translation performed by ProtoGen-MLIR v2\n";
    repeatChar('-', 20);
  }

  void repeatChar(char c, size_t reps){
    for(size_t i = 0; i < reps; i++)
      output << c;
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