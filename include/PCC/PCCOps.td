#ifndef PCC_OPS
#define PCC_OPS

//def PCCIntAttr : AttrConstraint<CPred<$_self.isa<IntegerAttr>()>, "PCC Integer Attribute">;

// TODO - remove
def FooOp : PCC_Op<"foo"> {
    let summary = "foo op";
    let description = [{}];
    let results = (outs PCCType:$results);
    let assemblyFormat = [{
        attr-dict `:` type($results)
    }];
}

def ConstantOp : PCC_Op<"constant"> {
    let summary = "define a global integer constant at the root module level";
    let description = [{
        Only to be used at the root level of a module. This operation will be used to represent
        constants in murphi, which can only be define at a top level.
    }];

    let arguments = (ins I64Attr:$val, StrAttr:$id);
    let results = (outs I64:$result);
    let assemblyFormat = [{attr-dict `:` type($result)}];

    let builders = [
            OpBuilderDAG<(ins "std::string":$id, "int64_t":$value), [{
                build($_builder,
                      $_state,
                      $_builder.getI64Type(),
                      $_builder.getI64IntegerAttr(value),
                      $_builder.getStringAttr(id)
                      );
    }]>
            ];
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

    let builders = [OpBuilderDAG<(ins
                    "StringRef":$cacheId,
                    "Type":$type,
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

    let builders = [OpBuilderDAG<(ins "StringRef":$dirId,
                    "Type":$type,
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

def ProcessOp : PCC_Op<"process", [FunctionLike]>{
    let summary = "Process handler operations";
    let description = "A Op which represents a particular action and events to be triggered on a cache or directory";

    // possibly have a SymbolNameAttr for name of the process
    // takes a type attribute to hold the type of the process
    let arguments = (ins SymbolNameAttr:$sym_name, TypeAttr:$type);
    // Single region with entry block
    let regions = (region AnyRegion:$body);

    let builders = [OpBuilderDAG<(ins
                    "StringRef":$name, "FunctionType":$type,
                    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs,
                    CArg<"ArrayRef<DictionaryAttr>", "{}">:$argAttrs)
  >];

    let extraClassDeclaration = [{
//        static ProcessOp create(Location loc, ProcessType type);
//        ProcessType getProcType();
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

    let parser = [{ return ::parseProcessOp(parser, result); }];
    let printer = [{ return ::print(*this, p); }];
}


def BreakOp : PCC_Op<"break", [Terminator]>{
    let summary = "default terminator op";
    let description = "A default terminator for terminating any block";
    let assemblyFormat = [{attr-dict}];
}


#endif // PCC_OPS