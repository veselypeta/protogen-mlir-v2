#include "FSM/Passes/Passes.h"
namespace {
#define GEN_PASS_REGISTRATION
#include "FSM/Passes/Passes.h.inc"
} // namespace

namespace mlir {
namespace fsm {
void registerAllFSMPasses() { registerPasses(); }

} // namespace fsm
} // namespace mlir