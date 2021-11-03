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

def EndStatesPass : Pass<"pcc-end-states", "ModuleOp"> {
    let summary = "For process ops with end states, adds set state ops";
    let description = [{
        Some Transaction in PCC can be specified with an optional end state

        i.e. in MESI the following transaction specifies a transaction from S->I
        without using explicit State update operations
        Process(S, Inv, I){
            msg = Resp(Inv_Ack, ID, Inv.src, cl);
            resp.send(msg);
        }

        The pass identifies such transactions and inserts set-state operations
    }];

    let constructor = "mlir::pcc::createEndStatesPass()";
}

def MutexOpsPass : Pass<"mutex-pass", "ModuleOp"> {
    let summary = "for stable compilation we add mutexes to the start and end of each transaction";
    let description = [{}];
    let constructor = "mlir::pcc::createMutexOpsPass()";
}


#endif // PCC_DIALECT_PASSES