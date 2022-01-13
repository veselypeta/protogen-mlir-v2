#ifndef PROTOGEN_DIALECT_FSM_FSMOPS_TD
#define PROTOGEN_DIALECT_FSM_FSMOPS_TD

include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

def HasCustomSSAName : DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>;


def MachineOp: FSMOp<"machine", [HasParent<"mlir::ModuleOp">,
        Symbol, SymbolTable, IsolatedFromAbove, NoTerminator, FunctionLike]>{
    let summary = "Define an FSM (typically a cache or directory controller)";
    let description = [{
    `fsm.machine` represents a finite-state machine
    }];
    let arguments = (ins StrAttr:$sym_name, TypeAttr:$type);
    let regions = (region SizedRegion<1>:$body);

    let builders = [OpBuilder<(ins
                    "StringRef":$name, "FunctionType":$type,
                    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
                    CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs)
    >];

    let extraClassDeclaration = [{
    /// Get the starting state of the machine i.e. the default state/ usually I


    private:
        /// This train needs access to the hooks defined below.
        friend class OpTrait::FunctionLike<MachineOp>;

        /// Returns the number of arguments. This is a hook for
        /// OpTrait::FunctionLike.
        unsigned getNumFuncArguments() { return getType().getInputs().size(); }

        /// Return the number of results. This is a hook for OpTrait::FunctionLike.
        unsigned getNumFuncResults() { return getType().getResults().size(); }

        /// Hook for OpTrait::FunctionLike, called after verifying that the 'type'
        /// attribute is present and checks if it holds a function type. Ensures
        /// getType, getNumFuncArguments, and getNumFuncResults can be called
        /// safely.
        LogicalResult verifyType(){
            auto type = getTypeAttr().getValue();
            if(!type.isa<FunctionType>())
                return emitOpError("requires '" + getTypeAttrName() +
                    "' attribute of function type");
            return success();
        }

    }];

    let parser = [{ return ::parse$cppClass(parser, result); }];
    let printer = [{ return ::print(*this, p); }];
    let verifier = [{ return ::verify$cppClass(*this); }];
}

def VariableOp : FSMOp<"variable", [HasParent<"MachineOp">, HasCustomSSAName,
      FirstAttrDerivedResultType]> {
  let summary = "Declare a variable in a state machine";
  let description = [{
    `fsm.variable` represents an internal variable in a state machine with an
    initialization value.
  }];

  let arguments = (ins AnyAttr:$initValue, StrAttr:$name);
  let results = (outs AnyType:$result);

  let assemblyFormat = [{ $name attr-dict `:` type($result) }];
}

def StateOp : FSMOp<"state", [HasParent<"MachineOp">, Symbol, SymbolTable, NoTerminator, SingleBlock]> {

    let summary = "Define a state of a machine";
    let description = [{
       `fsm.state` represents a state of a state machine.
    }];
    let arguments = (ins    SymbolNameAttr:$sym_name,
                            OptionalAttr<BoolAttr>:$isTransient,
                            OptionalAttr<SymbolRefAttr>:$prevTransition);

    let regions = (region SizedRegion<1>:$transitions);
    let assemblyFormat = [{
        $sym_name attr-dict `transitions` $transitions
    }];
    let extraClassDeclaration = [{
    public:
    Block *addEntryBlock(){
        assert(this->transitions().getBlocks().size() == 0 && "already added an entry block");
        auto block = new Block();
        this->transitions().push_back(block);
        return block;
    }
    }];
    let builders = [OpBuilder<(ins "StringRef":$state)>];
}

def TransitionOp : FSMOp<"transition", [HasParent<"StateOp">, Symbol, NoTerminator]>{
    let summary = "Define a transition of a state";
    let description = [{
        `fsm.transition` represents a transition of a fsm.
    }];
    let arguments = (ins SymbolNameAttr:$sym_name, OptionalAttr<SymbolRefAttr>:$nextState);

    let regions = (region SizedRegion<1>:$body);

    let assemblyFormat = [{
        attr-dict `action` $sym_name $body
    }];

    let extraClassDeclaration = [{
        public:
        Block *addEntryBlock(){
            assert(this->body().getBlocks().size() == 0 && "already added an entry block");
            auto block = new Block();
            this->body().push_back(block);
            return block;
        }

    }];
}

def MessageOp : FSMOp<"message", []>{
    let summary = "Constructs a message which can then be sent onto a network";
    let description = [{}];

    let arguments = (ins SymbolRefAttr:$msgType, StrAttr:$msgName);
    let results = (outs AnyType:$result);

}





#endif // PROTOGEN_DIALECT_FSM_FSMOPS_TD