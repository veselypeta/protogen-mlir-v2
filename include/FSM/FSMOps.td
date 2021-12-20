#ifndef PROTOGEN_DIALECT_FSM_FSMOPS_TD
#define PROTOGEN_DIALECT_FSM_FSMOPS_TD

include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"



def MachineOp: FSMOp<"machine", [HasParent<"mlir::ModuleOp">,
        Symbol, SymbolTable, IsolatedFromAbove, NoTerminator]>{
    let summary = "Define an FSM (typically a cache or directory controller)";
    let description = [{
    `fsm.machine` represents a finite-state machine
    }];
    let arguments = (ins StrAttr:$sym_name, TypeAttr:$type);
    let regions = (region SizedRegion<1>:$body);
}

def StateOp : FSMOp<"state", [HasParent<"MachineOp">, Symbol, NoTerminator]> {

    let summary = "Define a state of a machine";
    let description = [{
       `fsm.state` represents a state of a state machine.
    }];
    let arguments = (ins StrAttr:$sym_name);
    let regions = (region SizedRegion<1>:$transitions);
    let assemblyFormat = [{
        $sym_name attr-dict `transitions` $transitions
    }];
}

def TransitionOp : FSMOp<"transition", [HasParent<"StateOp">]>{
    let summary = "Define a transition of a state";
    let description = [{
        `fsm.transition` represents a transition of a fsm.
    }];

    let regions = (region SizedRegion<1>:$body);
}





#endif // PROTOGEN_DIALECT_FSM_FSMOPS_TD