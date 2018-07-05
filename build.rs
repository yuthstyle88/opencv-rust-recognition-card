extern crate cmake;

use cmake::Config;
use std::env;

fn main()
{
    let dst = Config::new("libfoo")    // The `libfoo` str here stands for the name of a source 
                                        // directory with native code equipped with CMake driven build.
                   .build();            // This runs the cmake to generate makefiles and build it all eventually

    // Same for C++ - to build the static libary from libfoo++ subdirectory,
    // but we don't need to consume its result. 
    Config::new("libfoo++").build();      

    // Now - emitting some cargo commands to build and link the lib. 
    // This turns to be common to both our libs, so we do it once.
    println!("cargo:rustc-link-search=native={}", dst.display());
    // Phase `foo` here stands for the library name (without lib prefix and without .a suffix)
    println!("cargo:rustc-link-lib=static=foo");    
    

    println!("cargo:rustc-link-lib=static=foo++");

    // C++ is bit more complicated, since platform specifics come to play
    let target  = env::var("TARGET").unwrap();
    if target.contains("apple")
    {
        println!("cargo:rustc-link-lib=dylib=c++");
    }
    else if target.contains("linux")
    {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
    else 
    {
        unimplemented!();
    }
}
