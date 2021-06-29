#include "translation/murphi/TranslatePCCToMurphi.h"
#include "PCC/PCCOps.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Translation.h"

#include <string>
using namespace mlir::pcc;

// private namespace for the implementation
namespace {

class MurphiTranslateImpl {
public:
  MurphiTranslateImpl(mlir::ModuleOp op, mlir::raw_ostream &output)
      : theModule{op}, output{output} {}

  mlir::LogicalResult translate() {
    generatePreamble();


    for(auto &op : getModuleBody().getOperations()){
      auto constOp = mlir::dyn_cast<ConstantOp>(op);
      if(constOp != nullptr){
        const std::string constId = constOp.id().str();
        const uint64_t constVal = constOp.val();
      }
    }



    return mlir::success();
  }

private:
  mlir::ModuleOp theModule;
  mlir::raw_ostream &output;

  void generatePreamble() {
    repeatChar('-', 20);
    output << '\n';
    repeatChar('-', 2);
    output << " Translation performed by ProtoGen-MLIR v2\n";
    repeatChar('-', 20);
    repeatChar('\n', 2);
  }

  void repeatChar(const char c, const size_t reps) const {
    for (size_t i = 0; i < reps; i++)
      output << c;
  }

  mlir::Block &getModuleBody(){
    return theModule.getOperation()->getRegion(0).front();
  }

  void processConstantOp(ConstantOp &constantOp){
    auto id = constantOp.id();
    auto val = constantOp.val();
    // Make some sort of constant op in murphi
    auto tmpl = "const " + id + " : " + std::to_string(val);
  }
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