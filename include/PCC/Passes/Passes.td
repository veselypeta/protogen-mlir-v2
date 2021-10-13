#ifndef PCC_DIALECT_PASSES
#define PCC_DIALECT_PASSES

include "mlir/Pass/PassBase.td"


def TransientStates : Pass<"pcc-trans-states", "ModuleOp"> {
    let summary = "Removes await blocks into individual transient states";
    let description = [{
        TODO
    }];
    let constructor = "mlir::pcc::createTransientStatesPass()";
}


#endif // PCC_DIALECT_PASSES