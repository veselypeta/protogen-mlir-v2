#ifndef PROTOGEN_FSM_DIALECT
#define PROTOGEN_FSM_DIALECT

include "mlir/IR/OpBase.td"

def FSMDialect : Dialect {
    let name = "fsm";
    let summary = "Types and operations for FSM dialect";
    let description = [{
        This dialect defined the `fsm` dialect, which is intended to represent
        finite-state machines.
    }];

    let cppNamespace = "::mlir::fsm";
}

// Base class for the types in this dialect
class FSMType<string name> : TypeDef<FSMDialect, name>{}

// base class for operations in this dialect.
class FSMOp<string mnemonic, list<OpTrait> traits = []> :
    Op<FSMDialect, mnemonic, traits>;

include "FSMTypes.td"
include "FSMOps.td"

#endif // PROTOGEN_FSM_DIALECT