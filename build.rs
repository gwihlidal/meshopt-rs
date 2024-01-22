use std::env;

fn main() {
    let mut build = cc::Build::new();

    build.include("src");

    // Add the files we build
    let source_files = [
        "vendor/src/allocator.cpp",
        "vendor/src/clusterizer.cpp",
        "vendor/src/indexcodec.cpp",
        "vendor/src/indexgenerator.cpp",
        "vendor/src/overdrawanalyzer.cpp",
        "vendor/src/overdrawoptimizer.cpp",
        "vendor/src/simplifier.cpp",
        "vendor/src/spatialorder.cpp",
        "vendor/src/stripifier.cpp",
        "vendor/src/vcacheanalyzer.cpp",
        "vendor/src/vcacheoptimizer.cpp",
        "vendor/src/vertexcodec.cpp",
        "vendor/src/vfetchanalyzer.cpp",
        "vendor/src/vfetchoptimizer.cpp",
    ];

    for source_file in &source_files {
        build.file(source_file);
    }

    let target = env::var("TARGET").unwrap();
    if target.contains("darwin") {
        build
            .flag("-std=c++11")
            .cpp_link_stdlib("c++")
            .cpp_set_stdlib("c++")
            .cpp(true);
    } else if target.contains("linux") || target.contains("windows-gnu") {
        build.flag("-std=c++11").cpp_link_stdlib("stdc++").cpp(true);
    }

    if target.starts_with("wasm32") {
        // In webassembly there's no stdlib, so we use
        // our own stripped down headers to provide the few
        // functions needed via LLVM intrinsics.
        build.flag("-isystem").flag("include_wasm32");
        // The Wasm backend needs a compatible ar
        // which will most likely be available under
        // this name on Windows, via manual LLVM install
        let host = env::var("HOST").unwrap();
        if host.contains("windows") {
            build.archiver("llvm-ar");
        }
    }

    build.compile("meshopt_cpp");

    generate_bindings("gen/bindings.rs");
}

#[cfg(feature = "generate_bindings")]
fn generate_bindings(output_file: &str) {
    let bindings = bindgen::Builder::default()
        .header("vendor/src/meshoptimizer.h")
        .rustfmt_bindings(true)
        .derive_debug(true)
        .impl_debug(true)
        .blocklist_type("__darwin_.*")
        .allowlist_function("meshopt.*")
        .trust_clang_mangling(false)
        .layout_tests(false)
        .size_t_is_usize(true)
        .generate()
        .expect("Unable to generate bindings!");

    bindings
        .write_to_file(std::path::Path::new(output_file))
        .expect("Unable to write bindings!");
}

#[cfg(not(feature = "generate_bindings"))]
fn generate_bindings(_: &str) {}
