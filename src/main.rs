extern { 
    // this tells we're declaring something living in the external library
    // `foo` and `foo++` stand here as names of the libraries (wihout lib prefix)

    // this is rustified prototype of the function from our C library
    #[link(name="foo", kind="static")]
    fn testcall(v: f32); 

    // this is rustified prototype of the function from our C++ library
    #[link(name="foo++", kind="static")]
    fn testcall_cpp(v: f32); 
}

fn main() {
    println!("Hello, world from Rust!");

    // now it's time to call the external function
    // In rust this comes via unsafe code block.
    unsafe { 
        testcall(3.14159); 
        testcall_cpp(3.14159); 
    };
}
