include_guard()

function(add_protogen_dialect dialect dialect_namespace)
    add_mlir_dialect(${ARGV})
    add_dependencies(protogen-headers MLIR${dialect}IncGen)
endfunction()