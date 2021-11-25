#ifndef PCC_OPS
#define PCC_OPS

def MsgConstrOp : PCC_Op<"msg_constr"> {
    let summary = "message constructor operation";
    let description = [{}];

    let arguments = (ins
            IDType:$src,
            IDType:$dst,
            TypeAttr:$mtype
            );
    let results = (outs PCCType:$results);
    let assemblyFormat = [{
        $src $dst attr-dict `:` type($results)
    }];
}

def StructAccessOp : PCC_Op<"struct_access"> {
    let summary = "access a field in a struct";
    let description =[{}];

    let arguments = (ins
            StructType:$input,
            StrAttr:$fieldId
            );

    let results = (outs PCCType:$results);
    let assemblyFormat = [{
       $input `[` $fieldId `]` attr-dict `:` type($input) `->` type($results)
    }];
}

def ConstantOp : PCC_Op<"constant", []> {
    let summary = "define a global integer constant at the root module level";
    let description = [{
        Only to be used at the root level of a module. This operation will be used to represent
        constants in murphi, which can only be define at a top level.
    }];

    let arguments = (ins I64Attr:$val, StrAttr:$id);
    let results = (outs I64:$result);
    let assemblyFormat = [{attr-dict `:` type($result)}];

    let builders = [
            OpBuilder<(ins "std::string":$id, "int64_t":$value), [{
                build($_builder,
                      $_state,
                      $_builder.getI64Type(),
                      $_builder.getI64IntegerAttr(value),
                      $_builder.getStringAttr(id)
                      );
    }]>
            ];
}

def InlineConstOp: PCC_Op<"inl_const", []>{
    let summary = "define an line constant";
    let description = [{}];

    let arguments = (ins AnyAttr:$value);
    let results = (outs AnyType:$result);

    let skipDefaultBuilders = 1;
    let builders = [
        OpBuilder<(ins "Attribute":$value)>,
        OpBuilder<(ins "StateType":$value)>
    ];
    let parser = [{ return ::parse$cppClass(parser, result); }];
    let printer = [{ return ::print(p, *this); }];
}

// PCC Op to declare a interconnect network (either ordered or unordered)
def NetDeclOp : PCC_Op<"net_decl">{
    let summary = "Declare global network";
    let description = [{
        Declare a global `net_decl` operation. Returns a NetworkType SSA value which can be used in
        other operations.
    }];
    let arguments = (ins StrAttr:$netId);
    let results = (outs NetworkType:$result);
    let assemblyFormat = [{
        attr-dict `:` type($result)
    }];
    // TODO - Add a method to get the type of network (ordered/unordered)
}

def CacheDeclOp : PCC_Op<"cache_decl">{
    let summary = "Declare the global type of a the cache(s)";
    let description = [{
        Declare the cache object types field field ids. Returns a SSA value which can be used in
        other operations.
    }];
    let arguments = (ins SymbolNameAttr:$id);
    let results = (outs PCCType:$result);
    let assemblyFormat = [{
        attr-dict `:` type($result)
    }];

    let builders = [OpBuilder<(ins
                    "StringRef":$cacheId,
                    "PCCType":$type,
                    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs
                    )>];
    let extraClassDeclaration = [{
        TypeAttr getFieldAttr(llvm::StringRef name){
            bool hasAttr = getOperation()->hasAttr(name);
            assert(hasAttr && "The cache does not posses a field with this attribute");
            return getOperation()->getAttr(name).dyn_cast<mlir::TypeAttr>();
        }
    }];
}

def DirectoryDeclOp : PCC_Op<"directory_decl">{
    let summary = "Declare a directory global type declaration";
    let description = [{
        Declare the directory global object which can be used by behavioural implementations.
    }];
    let arguments = (ins SymbolNameAttr:$id);
    let results = (outs PCCType:$result);
    let assemblyFormat = [{
        attr-dict `:` type($result)
    }];

    let builders = [OpBuilder<(ins "StringRef":$dirId,
                    "PCCType":$type,
                    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)
                            >];
    let extraClassDeclaration = [{
        TypeAttr getFieldAttr(llvm::StringRef name){
            bool hasAttr = getOperation()->hasAttr(name);
            assert(hasAttr && "The directory does not posses a field with this attribute");
            return getOperation()->getAttr(name).dyn_cast<mlir::TypeAttr>();
        }
    }];
}

def MsgDeclOp : PCC_Op<"msg_decl">{
    let summary = "Declare a global message type";
    let description = [{
        An op to declare a global message type that allows caches or directory to send messages of this type.
    }];
    let arguments = (ins SymbolNameAttr:$id);
    let results = (outs PCCType:$result);
    let assemblyFormat = [{
        attr-dict `:` type($result)
    }];
    let builders = [
            OpBuilder<(ins "StringRef":$m_type, "StructType":$type, CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)>
            ];
    let extraClassDeclaration = [{
        TypeAttr getFieldAttr(llvm::StringRef name){
            bool hasAttr = getOperation()->hasAttr(name);
            assert(hasAttr && "The Message type does not contain the attribute!");
            return getOperation()->getAttr(name).dyn_cast<mlir::TypeAttr>();
        }

        StringRef getMsgName() {
             return getOperation()->getAttr("id").cast<StringAttr>().getValue();
        }
    }];
}

def ProcessOp : PCC_Op<"process", [FunctionLike, NoTerminator]>{
    let summary = "Process handler operations";
    let description = "A Op which represents a particular action and events to be triggered on a cache or directory";

    // possibly have a SymbolNameAttr for name of the process
    // takes a type attribute to hold the type of the process
    let arguments = (ins SymbolNameAttr:$sym_name, TypeAttr:$type);
    // Single region with entry block
    let regions = (region AnyRegion:$body);

    let builders = [OpBuilder<(ins
                    "StringRef":$name, "FunctionType":$type,
                    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
                    CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs)
  >];

    let extraClassDeclaration = [{
//        static ProcessOp create(Location loc, ProcessType type);
//        ProcessType getProcType();
    bool hasNestedAwait();
    private:
        // This trait needs access to the hooks defined below.
        friend class OpTrait::FunctionLike<ProcessOp>;
        /// Returns the number of arguments. This is a hook for
        /// OpTrait::FunctionLike.
        unsigned getNumFuncArguments() {return getType().getInputs().size(); }

        /// Returns the number of results. This is a hook for OpTrait::FunctionLike.
        unsigned getNumFuncResults() {return getType().getResults().size();}

        /// Hook for OpTrait::FunctionLike, called after verifying that the 'type'
        /// attribute is present and checks if it holds a function type. Ensures
        /// getType, getNumFuncArguments, and getNumFuncResults can be called
        /// safely.
        LogicalResult verifyType() {
            auto type = getTypeAttr().getValue();
            if (!type.isa<FunctionType>())
                return emitOpError("requires '" + getTypeAttrName() +
                                   "' attribute of function type");
            return success();
        }
    }];

    let parser = [{ return ::parse$cppClass(parser, result); }];
    let printer = [{ return ::print(*this, p); }];
}


def BreakOp : PCC_Op<"break", [Terminator]>{
    let summary = "default terminator op";
    let description = "A default terminator for terminating any block";
    let assemblyFormat = [{attr-dict}];
}


def TransactionOp : PCC_Op<"await", [NoTerminator, NoRegionArguments, SingleBlock]>{
    let summary = "";
    let description = "";
    let regions = (region SizedRegion<1>:$body);

    let extraClassDeclaration = [{

    Block *addEntryBlock(){
      Block *block = new Block();
      this->body().push_back(block);
      return block;
    }
    }];

    let assemblyFormat = [{
        attr-dict `:` $body
    }];
}


def TransitionOp: PCC_Op<"when", [SingleBlockImplicitTerminator<"BreakOp">]>{
    let summary = "";
    let description = [{}];

    let arguments = (ins StrAttr:$guard, TypeAttr:$msgType);
    let regions = (region SizedRegion<1>:$body);

    let parser = [{ return ::parse$cppClass(parser, result); }];
    let printer = [{ return ::print(*this, p); }];

    let extraClassDeclaration = [{

    Block *addEntryBlock(){
      Block *block = new Block();
      this->body().push_back(block);
      block->addArgument(this->msgType());
      return block;
    }
    }];
}

def MsgSendOp: PCC_Op<"send">{
    let summary = "Sends a Msg on a network";
    let description = [{}];

    let arguments = (ins NetworkType:$network, StructType:$message);
    let assemblyFormat = [{
       attr-dict `:` $network`:`type($network) $message`:`type($message)
    }];

    let extraClassDeclaration = [{}];

}


def StateUpdateOp: PCC_Op<"update">{
    let summary = "Update a state variable of the FMS i.e. cache or directory";
    let description = [{
        pcc.update [ cl ] %cache %value
    }];

    let arguments = (ins StructType:$mach, AnyType:$value, StrAttr:$field);

    let parser = [{ return ::parse$cppClass(parser, result); }];
    let printer = [{ return ::print(*this, p); }];

}


def IfOp: PCC_Op<"if", [NoTerminator]>{
    let summary = "An op that represents a If then else control flow";
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


// MUTEX OPS
def AcquireOp: PCC_Op<"acquire">{
    let summary = "acquire a global mutex for all messages";
    let description = [{}];
    let assemblyFormat = [{attr-dict}];
}

def ReleaseOp: PCC_Op<"release">{
    let summary = "release the mutex lock acquired before";
    let description = [{}];
    let assemblyFormat = [{attr-dict}];
}


// Logical Ops
def LogicalAnd : PCC_Op<"and">{
    let summary = "Performs logical and on the two input operands";
    let description = [{}];

    let arguments = (ins I1:$lhs, I1:$rhs);
    let results = (outs I1:$result);

    let assemblyFormat = [{attr-dict `:` $lhs $rhs `->` type($result)}];

    let builders = [
        OpBuilder<(ins "Value":$lhs, "Value":$rhs), [{
            build(
                $_builder,
                $_state,
                $_builder.getI1Type(),
                lhs,
                rhs
            );
        }]>
    ];
}

def LogicalOr : PCC_Op<"or">{
    let summary = "Performs logical 'or' on the two input operands";
    let description = [{}];

    let arguments = (ins I1:$lhs, I1:$rhs);
    let results = (outs I1:$result);

    let assemblyFormat = [{attr-dict `:` $lhs $rhs `->` type($result)}];

    let builders = [
        OpBuilder<(ins "Value":$lhs, "Value":$rhs), [{
            build(
                $_builder,
                $_state,
                $_builder.getI1Type(),
                lhs,
                rhs
            );
        }]>
    ];
}


// Relational Operations
def EqualOp : PCC_Op<"eq">{
    let summary = "performs equality of the two operands";
    let description = [{}];

    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs I1:$result);

    let assemblyFormat = [{ attr-dict `:` $lhs`:`type($lhs) $rhs`:`type($rhs) `->` type($result)}];

    let builders = [
        OpBuilder<(ins "Value":$lhs, "Value":$rhs), [{
            build(
                $_builder,
                $_state,
                $_builder.getI1Type(),
                lhs,
                rhs
            );
        }]>
    ];
}

def LessOrEqualOp : PCC_Op<"lt_eq">{
    let summary = "performs less than or equal of the two operands";
    let description = [{}];

    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs I1:$result);

    let assemblyFormat = [{ attr-dict `:` $lhs`:`type($lhs) $rhs`:`type($rhs) `->` type($result)}];

    let builders = [
        OpBuilder<(ins "Value":$lhs, "Value":$rhs), [{
            build(
                $_builder,
                $_state,
                $_builder.getI1Type(),
                lhs,
                rhs
            );
        }]>
    ];
}


// Math ops

def AddOp : PCC_Op<"add">{
    let summary = "add two operands together";
    let description = [{}];

    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$result);
    let assemblyFormat = [{ attr-dict `:` $lhs`:`type($lhs) $rhs`:`type($rhs) `->` type($result)}];
}

def MultOp : PCC_Op<"mult">{
    let summary = "multiply two operands together";
    let description = [{}];

    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$result);
    let assemblyFormat = [{ attr-dict `:` $lhs`:`type($lhs) $rhs`:`type($rhs) `->` type($result)}];
}

def SubOp : PCC_Op<"sub">{
    let summary = "subtract two operands together";
    let description = [{}];

    let arguments = (ins AnyType:$lhs, AnyType:$rhs);
    let results = (outs AnyType:$result);
    let assemblyFormat = [{ attr-dict `:` $lhs`:`type($lhs) $rhs`:`type($rhs) `->` type($result)}];
}


#endif // PCC_OPS