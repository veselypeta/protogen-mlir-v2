#ifndef FSM_DIALECT_PASSES
#define FSM_DIALECT_PASSES

include "mlir/Pass/PassBase.td"

def StallingOptimizationPass: Pass<"fsm-stalling-optimization", "ModuleOp"> {
   let summary = "Optimizes a lock based protocol into an optimized lock-free stalling one";
   let description = [{
       TODO
   }];
   let constructor = "mlir::fsm::createStallingOptimizationPass()";
   let dependentDialects = ["mlir::fsm::FSMDialect"];
}

#endif // FSM_DIALECT_PASSES