#ifndef PROTOGEN_DIALECT_FSM_FSMTYPES_TD
#define PROTOGEN_DIALECT_FSM_FSMTYPES_TD


def FSMType : DialectType<FSMDialect, CPred<"$_self.isa<FSMType>()">,
    "FSMType", "::mlir::fsm::FSMType">;

// Simple Opaque Types
def IDType : DialectType<FSMDialect, CPred<"$_self.isa<IDType>()">, "id", "::mlir::fsm::IDType">,
    BuildableType<"IDType::get($_builder.getContext())">;

def DataType : DialectType<FSMDialect, CPred<"$_self.isa<DataType>()">,"data", "::mlir::fsm::DataType">,
    BuildableType<"DataType::get($_builder.getContext())">;

def MsgType : DialectType<FSMDialect, CPred<"$_self.isa<MsgType>()">, "msg", "::mlir::fsm::MsgType">,
    BuildableType<"MsgType::get($_builder.getContext())">;

def StateType : DialectType<FSMDialect, CPred<"$_self.isa<StateType>()">, "msg", "::mlir::fsm::StateType">,
    BuildableType<"StateType::get($_builder.getContext())">;

def RangeType : DialectType<FSMDialect, CPred<"$_self.isa<RangeType>()">, "range", "::mlir::fsm::RangeType">,
        BuildableType<"RangeType::get($_builder.getContext())">;
#endif // PROTOGEN_DIALECT_FSM_FSMTYPES_TD
