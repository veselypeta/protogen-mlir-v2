#ifndef PCC_OPS
#define PCC_OPS

def FooOp : PCC_Op<"foo"> {
    let summary = "foo op";
    let description = [{}];
    let results = (outs PCCType:$results);
    let assemblyFormat = [{
        attr-dict `:` type($results)
    }];
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

    // TODO -- add builder to allow for easy Op construction
}

def ProcessOp : PCC_Op<"process", [FunctionLike]>{
    let summary = "Process handler operations";
    let description = "A Op which represents a particular action and events to be triggered on a cache or directory";

    // possibly have a SymbolNameAttr for name of the process
    // takes a type attribute to hold the type of the process
    let arguments = (ins SymbolNameAttr:$sym_name, TypeAttr:$type);
    // Single region with entry block
    let regions = (region AnyRegion:$body);

    let extraClassDeclaration = [{
        static ProcessOp create(Location loc, ProcessType type);
        ProcessType getProcType();
        ProcessType getType(){return getProcType();}
    private:
        // This trait needs access to the hooks defined below.
        friend class OpTrait::FunctionLike<ProcessOp>;
        /// Returns the number of arguments. This is a hook for
        /// OpTrait::FunctionLike.
        unsigned getNumFuncArguments() { return type().cast<ProcessType>().getNumInputs(); }

        /// Returns the number of results. This is a hook for OpTrait::FunctionLike.
        unsigned getNumFuncResults() { return type().cast<ProcessType>().getNumResults(); }

        /// Hook for OpTrait::FunctionLike, called after verifying that the 'type'
        /// attribute is present and checks if it holds a function type. Ensures
        /// getType, getNumFuncArguments, and getNumFuncResults can be called
        /// safely.
        LogicalResult verifyType() {
            auto type = getTypeAttr().getValue();
            if (!type.isa<ProcessType>())
                return emitOpError("ProcessOp Requires a Process Type");
            return success();
        }
    }];

    let printer = [{ return ::print(*this, p); }];
}


#endif // PCC_OPS