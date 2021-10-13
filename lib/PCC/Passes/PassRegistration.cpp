#include "PCC/Passes/Passes.h"
#include "mlir/Pass/Pass.h"

namespace {
#define GEN_PASS_REGISTRATION
#include "PCC/Passes/Passes.h.inc"
} // namespace


void mlir::pcc::initAllPCCPasses(){
  registerPasses();
}