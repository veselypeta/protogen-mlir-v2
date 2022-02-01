#ifndef PROTOGEN_DIALECT_FSM_FSMOPS_TD
#define PROTOGEN_DIALECT_FSM_FSMOPS_TD

include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

def HasCustomSSAName : DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>;


def MachineOp: FSMOp<"machine", [HasParent<"mlir::ModuleOp">,
        Symbol, SymbolTable, NoTerminator, FunctionLike]>{
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
}

def VariableOp : FSMOp<"variable", [HasParent<"MachineOp">, HasCustomSSAName,
      FirstAttrDerivedResultType]> {
  let summary = "Declare a variable in a state machine";
  let description = [{
    `fsm.variable` represents an internal variable in a state machine with an
    initialization value.
  }];

  let arguments = (ins OptionalAttr<AnyAttr>:$initValue, StrAttr:$name);
  let results = (outs AnyType:$result);

  let assemblyFormat = [{ $name attr-dict `:` type($result) }];
}

def UpdateOp : FSMOp<"update", [/*SameTypeOperands = implemented in verifier*/]>{
    let summary = "Update a variable in the state machine";
    let description = [{}];

    let arguments = (ins AnyType:$variable, AnyType:$value);


    let assemblyFormat = [{ attr-dict $variable `,` $value `:` type($variable) `,` type($value) }];
    let extraClassDeclaration = [{
        VariableOp getVariable();
    }];
    let verifier = [{return ::verify$cppClass(*this); }];
}

def ConstOp : FSMOp<"constant", []>{
    let summary = "Define a constant that can be used within pcc";
    let description = [{ I mostly just use this for creating a state type constant }];

    let arguments = (ins AnyAttr:$value);
    let results = (outs AnyType:$output);
    let assemblyFormat = [{
        attr-dict `:` type($output)
    }];
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

def BreakOp : FSMOp<"break", [Terminator]>{
    let assemblyFormat = [{ attr-dict }];
}

def TransitionOp : FSMOp<"transition", [HasParent<"StateOp">, Symbol, NoTerminator, FunctionLike]>{
    let summary = "Define a transition of a state";
    let description = [{
        `fsm.transition` represents a transition of a fsm.
    }];
    let arguments = (ins    SymbolNameAttr:$sym_name,
                            TypeAttr:$type,
                            OptionalAttr<SymbolRefAttr>:$nextState);

    let regions = (region SizedRegion<1>:$body);

    //let assemblyFormat = [{
    //    attr-dict `action` $sym_name $body
    //}];

    let builders = [
        OpBuilder<(ins
                        "StringRef":$name, "FunctionType":$type, "SymbolRefAttr":$nextState,
                        CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
                        CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs)>
    ];

    let extraClassDeclaration = [{

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
}

def MessageOp : FSMOp<"message", []>{
    let summary = "Constructs a message which can then be sent onto a network";
    let description = [{}];

    let arguments = (ins SymbolRefAttr:$msgType, StrAttr:$msgName, Variadic<AnyType>:$inputs);
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
    attr-dict $msgType $msgName ($inputs^ `:` type($inputs))? `->` type(results)
    }];

    let verifier = [{ return ::verify$cppClass(*this); }];
}

def AccessOp : FSMOp<"access", []>{
    let summary = "Returns a reference to a field within a message";
    let description = [{}];

    let arguments = (ins AnyType:$msg, StrAttr:$memberId);
    let results = (outs AnyType:$value);

    let assemblyFormat = [{
        attr-dict $msg`:`type($msg) `->` type($value)
    }];
    let verifier = [{ return ::verify$cppClass(*this); }];
}

def NOPOp : FSMOp<"nop", []>{
    let summary = "An operations the represents a No Op (NOP)";
    let description = [{}];
    let assemblyFormat = [{ attr-dict }];
}

def ReferenceOp : FSMOp<"ref", []>{
    let summary = "Returns a reference for either cache or directory";
    let description = [{}];

    let arguments = (ins SymbolRefAttr:$reference);
    let results = (outs IDType:$result);
    let assemblyFormat = [{ attr-dict $reference}];

}

def IfOp: FSMOp<"if", [NoTerminator, SingleBlock, NoRegionArguments]>{
    let summary = "if-then-else operation";
    let description = [{}];

    let arguments = (ins I1:$condition);
    let regions = (region SizedRegion<1>:$thenRegion, AnyRegion:$elseRegion);

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins "Value":$cond, "bool":$withElseRegion)>
    ];

    let extraClassDeclaration = [{
    Block *thenBlock();
    Block *elseBlock();
    }];

    let parser = [{ return ::parse$cppClass(parser, result); }];
    let printer = [{ return ::print(*this, p); }];
}

def NetworkOp : FSMOp<"network", [HasParent<"mlir::ModuleOp">, HasCustomSSAName]>{
    let summary = "Defines a global network onto which messages can be sent";
    let description = [{ TODO -- }];
    let arguments = (ins StrAttr:$ordering, SymbolRefAttr:$sym_name);
    let results = (outs NetworkType:$net);
    let verifier = [{ return ::verify$cppClass(*this); }];
    let assemblyFormat = [{
        attr-dict $sym_name $ordering
    }];
}

def SendOp : FSMOp<"send", []>{
    let summary = "Sends a message onto a network";
    let description = [{ TODO }];
    let arguments = (ins NetworkType:$network, MsgType:$message);
    let assemblyFormat = [{
        attr-dict $network $message
    }];
}

def DeferMsg : FSMOp<"defer", []>{
    let summary = "Pushes a message to a buffer to be sent at a later point";
    let description = [{ TODO }];
    let arguments = (ins MsgType:$message);
    let assemblyFormat = [{
        attr-dict $message
    }];
}

def SendDeferMsg : FSMOp<"defer_send", []>{
    let summary = "Sends a deferred message from the buffer";
    let description = [{ Deferred messages are always sent on the response network }];
    let arguments = (ins MsgType:$message);
    let assemblyFormat = [{ attr-dict $message }];
}

def MessageDecl : FSMOp<"m_decl", [SymbolTable, Symbol, NoTerminator, HasParent<"::mlir::ModuleOp">, SingleBlock]>{
    let summary = "Declares a message type";
    let description = [{ TODO }];
    let arguments = (ins SymbolNameAttr:$sym_name);
    let regions = (region SizedRegion<1>:$body);
    let assemblyFormat = [{
        $sym_name attr-dict `decls` $body
    }];
    let extraClassDeclaration = [{
    Block *addEntryBlock(){
        assert(this->body().getBlocks().size() == 0 && "already added an entry block");
        auto block = new Block();
        this->body().push_back(block);
        return block;
    }
    }];
}

def MessageVariable : FSMOp<"m_var", [Symbol, HasParent<"::mlir::fsm::MessageDecl">, HasCustomSSAName]>{
    let summary = "Defines a variable type as part of a Message Decl";
    let description = [{ TODO }];
    let arguments = (ins SymbolNameAttr:$sym_name);
    let results = (outs AnyType:$result);
    let assemblyFormat = [{
        attr-dict $sym_name `:` type($result)
    }];

}





#endif // PROTOGEN_DIALECT_FSM_FSMOPS_TD