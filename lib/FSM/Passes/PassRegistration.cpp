#include "FSM/Passes/Passes.h"
#include "PassDetail.h"
namespace {
#define GEN_PASS_REGISTRATION
#include "FSM/Passes/Passes.h.inc"
} // namespace

namespace mlir {
namespace fsm {
void registerAllFSMPasses() { registerPasses(); }
void registerAllFSMPipelines(){
  registerFullOptimizationPipeline();
}

} // namespace fsm
} // namespace mlir