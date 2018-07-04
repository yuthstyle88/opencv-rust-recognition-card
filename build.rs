extern crate cmake;
use cmake::Config;

fn main()
{
    let dst = Config::new("libfoo") // The `libfoo` str here stands for the name of a source 
                                    // directory with native code equipped with CMake driven build.
                    .build();       // This runs the cmake to generate makefiles and build it all eventually

    // now - emitting some cargo commands to build and link the lib
    println!("cargo:rustc-link-search=native={}", dst.display());
    // Phase `foo` here stands for the library name (without lib prefix and without .a suffix)
    println!("cargo:rustc-link-lib=static=foo");    
}
