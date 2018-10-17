extern crate cc;

//use std::env;

fn main() {
    let mut build = cc::Build::new();

    build.include("src");

    // Add the files we build
    let source_files = [
        "vendor/src/indexcodec.cpp",
        "vendor/src/indexgenerator.cpp",
        "vendor/src/overdrawanalyzer.cpp",
        "vendor/src/overdrawoptimizer.cpp",
        "vendor/src/simplifier.cpp",
        "vendor/src/stripifier.cpp",
        "vendor/src/vcacheanalyzer.cpp",
        "vendor/src/vcacheoptimizer.cpp",
        "vendor/src/vertexcodec.cpp",
        "vendor/src/vfetchanalyzer.cpp",
        "vendor/src/vfetchoptimizer.cpp",
    ];

    for source_file in &source_files {
        build.file(&source_file);
    }
    
    //let target = env::var("TARGET").unwrap();
    /*if target.contains("darwin") {
        build
            .flag("-std=c++11")
            .cpp(true)
            .cpp_link_stdlib("c++")
            .cpp_set_stdlib("c++");
    }*/

    build.compile("meshopt_cpp");
}