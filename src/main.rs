
// this tells we're declaring something living in the external library
// `foo` stands here as a name of the library (wihout lib prefix)
#[link(name="foo", kind="static")]
extern { 
    // this is rustified prototype of the function from our C library
    fn testcall(v: f32); 
}

fn main() {
    println!("Hello, world from Rust!");

    // now it's time to call the external function
    // In rust this comes via unsafe code block.
    unsafe { 
        testcall(3.14159); 
    };
}
