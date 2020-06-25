extern crate core;
extern crate opencv;
extern crate image;
extern crate libm;
extern crate glam;
extern {
    // this is rustified prototype of the function from our C++ library
    #[link(name="libhelper", kind="dynamic")]
    fn auto_close_line(img: Mat)-> Mat;
    fn test_image(img: Mat)-> Mat;
}

use opencv::{prelude::*, videoio, highgui, types};
use opencv::imgcodecs::{imread, IMREAD_COLOR, IMREAD_GRAYSCALE, imwrite};
use opencv::imgproc::{COLOR_BGR2GRAY, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, cvt_color, find_contours, line, LINE_AA, min_area_rect, draw_contours, THRESH_BINARY, threshold, warp_affine, get_rotation_matrix_2d, INTER_LINEAR, hough_lines_p, resize, rectangle, THRESH_BINARY_INV, gaussian_blur, bounding_rect, RETR_LIST, RETR_TREE, THRESH_OTSU, morphology_ex, RETR_CCOMP, canny, CHAIN_APPROX_NONE, contour_area, FILLED, put_text, COLOR_BGR2RGB, morphology_default_border_value};
use opencv::core::{Point, Mat, ToInputArray, MatTrait, InputArray, RotatedRect, MatTraitManual, Point2f, Scalar, Point2i, RotatedRectTrait, compare, CMP_GT, Size_, Point_, BORDER_CONSTANT, rotate, no_array, bitwise_not, CV_PI, CV_8UC1, Rect2f, Point3f, Size, Rect, Size2f, ToOutputArray, sort, sort_idx, Moments, absdiff, min, BORDER_DEFAULT, add_weighted, Vector, Vec3b};
use opencv::highgui::{imshow, wait_key, named_window};
use opencv::types::{VectorOfMat, VectorOfVec4i, VectorOfPoint, VectorOfPoint2f, VectorOfRect};
use std::borrow::{BorrowMut, Borrow};
use libm::atan2;
use image::imageops::crop;
use std::fs::copy;
use std::alloc::handle_alloc_error;
use std::convert::TryFrom;
use std::{env, thread};
use glam::Quat;
use opencv::videoio::{VideoCapture, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT};
use std::time::Duration;
use opencv::objdetect::CASCADE_SCALE_IMAGE;
use core::fmt;
use opencv::ximgproc::get_structuring_element;
use opencv::sys::cv_ml_ANN_MLP_create;

#[derive(Debug)]
pub struct Cards {
    card: Vec<(Mat, i32, f32)>
}

#[derive(Debug, PartialEq, Copy, Clone)]
enum CardLabel {
    Clubs,
    Diamonds,
    Hearts,
    Spades,
}

#[derive(Debug)]
pub struct Card {
    id: u8,
    name: String,
    location: u8,
}

impl fmt::Display for CardLabel {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl Cards {
    fn new() -> Self {
        Self { card: vec![] }
    }
    fn add(&mut self, card: (Mat, i32, f32)) {
        self.card.push(card)
    }
}

impl From<usize> for CardLabel {
    fn from(id: usize) -> CardLabel {
        let rs = match id {
            1 => CardLabel::Clubs,
            2 => CardLabel::Diamonds,
            3 => CardLabel::Hearts,
            4 => CardLabel::Spades,
            _ => CardLabel::Spades
        };
        rs
    }
}

const CORNER_WIDTH: i32 = 21;
const CORNER_HEIGHT: i32 = 60;
const SUIT_WIDTH: i32 = 70;
const SUIT_HEIGHT: i32 = 100;

fn split_card(src: &Mat, card: &mut Cards) {
    // imshow("010 CARD SRC  ", &src);
    // let _key = wait_key(0);
    let temp = no_array().unwrap();
    let mut dst = VectorOfMat::new();
    //bitwise_not(&src, &mut dst, &temp);
    let zero_offset = Point::new(7, 20);
    let mut vec = VectorOfMat::new();
    let mut gray_img = Mat::default().unwrap();
    cvt_color(&src, &mut gray_img, COLOR_BGR2GRAY, 0).unwrap();
    threshold(&gray_img.clone().unwrap(), &mut gray_img, 50., 255., THRESH_BINARY | THRESH_OTSU);
    //canny(&gray_img,&mut dst,100.,200.,3,false);

    let thres_img = gray_img.clone().unwrap();


    match find_contours(&thres_img, &mut vec, RETR_LIST, CHAIN_APPROX_NONE, zero_offset) {
        Ok(_ok) => {
            println!("[OK] find contours");
        }
        Err(error) => {
            println!("[KO] find contours: {}", error);
        }
    };
    //let sort_img = vec.clone().unwrap();

    // let mut xxx = src.clone().unwrap();
    let mut idx = 0;
    // let maxresult: i32 = 2147483647;
    //let mut hierachy = VectorOfMat::new();
    sort(&vec, &mut dst, 0);

    for cnt in vec.iter() {
        //dbg!(&cnt);

        let area = contour_area(&cnt, false).unwrap();

        if area < 30000f64 {
            continue;
        }
        let _min_area_rect = min_area_rect(&cnt).unwrap();
        let angle = _min_area_rect.angle();
        let mut roi = bounding_rect(&cnt).unwrap();
        roi.height = roi.height + 15;
        roi.width = roi.width + 15;
        roi.x = roi.x - 20;
        roi.y = roi.y - 20;


        let out = Mat::roi(&src, roi).unwrap();
        // imshow("CARD  000 SRC  ", &out);
        //let _key = wait_key(0);


        let id = get_card_location(roi.x, roi.y);
        println!("id {} ", id);
        let pair = (out, id,angle);
        card.add(pair);
        //copy from contour  CROP
        idx += 1;
    }
}

fn split_rank_suit(src: &Mat) -> (Mat, Mat) {
    let mut rank_img = Mat::default().unwrap();
    let mut suit_img = Mat::default().unwrap();
    let mut is_first = true;
    let mut vec_dst = VectorOfMat::new();

    let zero_offset = Point::new(0, 0);
    let mut vec = VectorOfMat::new();
    match find_contours(&src, &mut vec, RETR_LIST, CHAIN_APPROX_SIMPLE, zero_offset) {
        Ok(_ok) => {
            println!("[OK] find contours");
        }
        Err(error) => {
            println!("[KO] find contours: {}", error);
        }
    };

    let mut last_rank_y = 0;
    let mut area_selected:f64 = 1000.;
    let mut idx = 0;

    // Find suit image
    for cnt in vec.iter(){
        let area = contour_area(&cnt, false).unwrap();
        idx = idx + 1;
        // println!(">>> {} area -> {}", idx, area);
        if area < area_selected {
            continue;
        }
        // println!(">>> {} area selected -> {}", idx,  area);
        let mut roi = bounding_rect(&cnt).unwrap();
        let out = Mat::roi(&src, roi).unwrap();

        if true == is_first {
            suit_img = out;
            is_first = false;
            // println!(">>> {} suit -> {}", idx, area);
        }else {
            rank_img = out;
            // println!(">>> {} rank -> {}", idx, area);
        }
    }

    (rank_img, suit_img)
}

fn get_card_location(x: i32, y: i32) -> i32 {

    println!("get_card_location() -> x= {}, y={}",x,y);
    let pair = (x, y);
    let rs = match pair {
        (x, y) if x > 800 && y > 300 => 1,
        (x, y) if x > 700 && y > 300 => 2,
        (x, y) if x > 300 && y > 300 => 3,
        (x, y) if x > 100 && y > 300 => 4,
        (x, y) if x > 800 && y > 95 => 5,
        (x, y) if x > 100 && y > 95 => 6,
        _ => 0,
    };
    rs
}

fn get_card_name(rank: usize, suit: usize, location: u8, count_red: i32) -> Card {
    let suit = CardLabel::from(suit);


    let mut suit_name = suit ;
    // println!(": count_red -> {}", count_red);
    // println!(": suit -> {}", suit.to_string());
    //
    // if suit == CardLabel::Hearts && 1000 > count_red {
    //     suit_name = CardLabel::Spades;
    // } else if suit == CardLabel::Spades && 1000 < count_red { // is red
    //     suit_name = CardLabel::Hearts;
    // }
    // println!(": suit_name -> {}", suit_name.to_string());

    Card { id: rank as u8, name: suit_name.to_string(), location}
}


fn rankMatcher(rnk: Mat) -> usize {
    let mut dst = Mat::default().unwrap();
    // let mut qualities = Vec::with_capacity(14);
    let mut qualities = vec![0; 13];
    let mut minindex = 0;
    let mut rnk_resized = Mat::default().unwrap();

    let file = vec!["Card_Imgs/Ranks/1.jpg",
                    "Card_Imgs/Ranks/2.jpg",
                    "Card_Imgs/Ranks/3.jpg",
                    "Card_Imgs/Ranks/4.jpg",
                    "Card_Imgs/Ranks/5.jpg",
                    "Card_Imgs/Ranks/6.jpg",
                    "Card_Imgs/Ranks/7.jpg",
                    "Card_Imgs/Ranks/8.jpg",
                    "Card_Imgs/Ranks/9.jpg",
                    "Card_Imgs/Ranks/10.jpg",
                    "Card_Imgs/Ranks/11.jpg",
                    "Card_Imgs/Ranks/12.jpg",
                    "Card_Imgs/Ranks/13.jpg",
    ];

    // dbg!(&qualities);

    for i in 0..13 {
        let mut match_quality = 0;
        let mut dst = Mat::default().unwrap();
        let img = imread(file[i], IMREAD_GRAYSCALE).unwrap();

        resize(&rnk.clone().unwrap(), &mut rnk_resized, Size { width: img.cols(), height: img.rows() }, 0., 0., INTER_LINEAR);
        absdiff(&img, &rnk_resized, &mut dst);

        // display_picture_and_wait("absdiff", &dst);

        // Count the pixel
        for y in 0..dst.rows() {
            for x in 0..dst.cols() {
                if dst.at_pt::<u8>(Point::new(x, y)).unwrap() == &255u8 {
                    match_quality += 1;
                }
            }
        }
        qualities[i] = match_quality;
    }
    let mut min = qualities[0];

    for j in 0..13 {
        if qualities[j] < min {
            min = qualities[j];
            minindex = j;
        }
    }
    minindex += 1;
    println!("Minimum White Pixel on Position: {}", minindex);
    let file_param :Vector<i32> = Vector::new();
    let filename_str = format!("Card_Imgs/Ranks/{}_v.jpg",(minindex));
    imwrite(filename_str.as_ref(), &rnk_resized, &file_param);
    minindex
}

fn suitMatcher(suit: Mat) -> usize {
    let mut dst = Mat::default().unwrap();
    // let mut qualities = Vec::with_capacity(14);
    let mut qualities = vec![0; 4];
    let mut minindex = 0;
    let mut suit_resized = Mat::default().unwrap();

    let file = vec!["Card_Imgs/Suits/Clubs.jpg",
                    "Card_Imgs/Suits/Diamonds.jpg",
                    "Card_Imgs/Suits/Hearts.jpg",
                    "Card_Imgs/Suits/spades.jpg",
    ];

    // dbg!(&qualities);

    for i in 0..4 {
        let mut match_quality = 0;
        let mut dst = Mat::default().unwrap();
        let img = imread(file[i], IMREAD_GRAYSCALE).unwrap();

        resize(&suit.clone().unwrap(), &mut suit_resized, Size { width: img.cols(), height: img.rows() }, 0., 0., INTER_LINEAR);
        absdiff(&img, &suit_resized, &mut dst);

        // imshow("Suit matching", &dst);
        // let _key = wait_key(0);

        // Count the pixel
        for y in 0..dst.rows() {
            for x in 0..dst.cols() {
                if dst.at_pt::<u8>(Point::new(x, y)).unwrap() == &255u8 {
                    match_quality += 1;
                }
            }
        }
        qualities[i] = match_quality;
    }
    let mut min = qualities[0];

    for j in 0..4 {
        if qualities[j] < min {
            min = qualities[j];
            minindex = j;
        }
    }
    minindex += 1;
    println!("Minimum White Pixel on Position: {}", minindex);
    let file_param :Vector<i32> = Vector::new();
    let filename_str = format!("Card_Imgs/Suits/{}_v.jpg",(minindex));
    imwrite(filename_str.as_ref(), &suit_resized, &file_param);
    // println!("Saved file.");
    minindex
}


fn rotate_image90(src: &Mat) -> Mat {
    let size = src.size().unwrap();
    let width = size.height;
    let height = size.width;
    let center = Point2f::new(((size.width - 5) / 2) as f32, ((size.height + 70) / 2) as f32);
    let shape = Size_::new(width, height);
    let mut new_img = Mat::default().unwrap();
    let rot = get_rotation_matrix_2d(center, 90f64, 1f64);
    match rot {
        Ok(r) => {
            r.at::<i32>(-10);
            r.at::<i32>(-10);

            warp_affine(src, new_img.borrow_mut(), &r, shape, INTER_LINEAR, BORDER_CONSTANT, Scalar::default());
        }
        _ => {}
    }

    new_img
}

fn rotate_image(src: &Mat, angle: f64) -> Mat {

    let size = src.size().unwrap();
    let width = size.width;
    let height = size.height;
    let center = Point2f::new(((width - 1) / 2) as f32, ((height - 1) / 2) as f32);

    let rs = sub_image(&src, center, angle, width, height);

    rs
}

fn crop_card(src: &Mat, crop_size: f64) -> Mat {

    let mut img_cc = Mat::default().unwrap();
    let mut img_ts = Mat::default().unwrap();
    let mut img_cropped = Mat::default().unwrap();
    let is_show = false;

    process_img_gray(&src, &mut img_cc, is_show);
    process_img_threshold(&img_cc,&mut img_ts,  50., 255., is_show);
    let img_cropped = process_crop_by_img (src,  &img_ts, is_show);

    // let _result = imshow("crop_card cropped", &img_cropped);
    // let _key = wait_key(0);

    img_cropped
}

fn card_id(mut card: Mat, id: u8) -> Card {
    // dbg!(&card);
    let is_show = false;

    let is_big_card = chk_big_card(&card);
    println!(">>> is_big_card -> {}", is_big_card);
    let mut x = 12;
    // Cropped corner
    let mut corner_width = CORNER_WIDTH;
    if false == is_big_card {
        corner_width = CORNER_WIDTH;
        x = x + 5;
    }

    // display_picture_and_wait("before cropped", &card);
    let img_cropped = process_crop_img_by_size (&card,  x, 10, corner_width, CORNER_HEIGHT, is_show);
    let count_red = get_count_red(&img_cropped);

    // Resize image
    let mut img_corner = Mat::default().unwrap();
    let mut img_bo = Mat::default().unwrap();
    let mut img_gray = Mat::default().unwrap();
    let mut img_aw = Mat::default().unwrap();
    let mut img_ts = Mat::default().unwrap();

    process_img_resize(&img_cropped, &mut img_corner, 4, is_show);
    process_img_bitwise_not(&img_corner, &mut img_bo, is_show);
    process_img_gray(&img_bo, &mut img_gray, is_show);
    process_img_add_weighted(&img_gray, &mut img_aw,  is_show);
    process_img_threshold(&img_gray, &mut img_ts, 90., 255., is_show);

    // display_picture_and_wait("card_id conner_top_left", &img_ts);

    let (rank_img, suit_img) = split_rank_suit(&img_ts);
    // display_picture_and_wait("rank_img", &rank_img);
    // display_picture_and_wait("suit_img", &suit_img);

    let rank_result = rankMatcher(rank_img);
    println!(">>> rank_result -> {}", rank_result);
    let suit_result = suitMatcher(suit_img);
    println!(">>> suit_result -> {}", suit_result);

    get_card_name(rank_result, suit_result, id, count_red)
}

fn recognition_card(in_img: &Mat) -> Vec<Card> {
    let mut cards = Cards::new();
    split_card(&in_img, &mut cards);
    let mut img_count = 0;
    let mut res = vec![];
    for card_tuple in cards.card.into_iter() {
        let (mut card_in, id,mut angle) = card_tuple;
        let file_param :Vector<i32> = Vector::new();
        let fname = format!("test_{}.png",id);
        imwrite(&fname, &card_in, &file_param);
        let width = card_in.cols();
        let height = card_in.rows();
        if width > height {
            card_in = rotate_image90(&card_in);
        }
        let mut add_angle = 90.;
        if angle  < -50.  {angle  =  add_angle + angle;} else {angle = angle - 1.0};

        //let angle = find_angle(&card_in);
        println!(" {}. recognition_card->angle : {}", img_count, angle);

        display_picture_and_wait("card_in" , &card_in);
        let rotate_img = rotate_image(&card_in, angle as f64);
        display_picture_and_wait("rotate_img" , &rotate_img);
        let croped_img = crop_card(&rotate_img, 10000.);
        display_picture_and_wait("croped_img" , &croped_img);
        let card_result = card_id(croped_img, id as u8);

        res.push(card_result);

        // let _result = imshow("window_name", &quad);
        //let _key = wait_key(0);
    }
    res
}

fn run() -> opencv::Result<()> {
    let window = "video capture";
    named_window(window, 1)?;
    #[cfg(feature = "opencv-32")]
        let mut cam = VideoCapture::new_default(1)?;  // 0 is the default camera
    #[cfg(not(feature = "opencv-32"))]
        let mut cam = VideoCapture::new(1, videoio::CAP_ANY)?;  // 0 is the default camera
    cam.set(CAP_PROP_FRAME_WIDTH, 1280.);
    cam.set(CAP_PROP_FRAME_HEIGHT, 720.);
    let opened = VideoCapture::is_opened(&cam)?;
    if !opened {
        panic!("Unable to open default camera!");
    }
    loop {
        let mut frame = Mat::default()?;
        cam.read(&mut frame)?;
        if frame.size()?.width == 0 {
            thread::sleep(Duration::from_secs(50));
            continue;
        }
        let mut gray = Mat::default()?;
        /* cvt_color(
             &frame,
             &mut gray,
             COLOR_BGR2RGB,
             0
         )?;*/

        imshow(window, &frame)?;
        if wait_key(10)? > 0 {
            /*let mut increase = Mat::default()?;
            resize(
                &frame,
                &mut increase,
                Size {
                    width: 1280,
                    height: 720
                },
                0.25f64,
                0.25f64,
                INTER_LINEAR
            )?;
*/
            //dbg!(&increase);
            // recognition_card(&frame);
            let file_param :Vector<i32> = Vector::new();
            imwrite("test.png", &frame, &file_param);
            let card_dataset = recognition_card(&frame);
            dbg!(card_dataset);
            continue;
        }
    }
    Ok(())
}


fn main() {

    // run().unwrap();

    let filename = format!("src/{}", "sp.png");

    let mut in_img = match imread(&filename, IMREAD_COLOR) {
        Ok(ok) => ok,
        Err(error) => {
            panic!("Fatal ERROR reading the image : {}. Is this file exist? Have you the right to read it? Is it empty? . Error : {:?}", filename, error)
        }
    };

    display_picture_and_wait("main()", &in_img);

    let card_dataset = recognition_card(&in_img);
    dbg!(card_dataset);



    // let filename = format!("src/{}", "sp003.png");

    // let in_img = imread(&filename, IMREAD_COLOR).unwrap();
    // display_picture_and_wait("main() 1", &in_img);
    // let x = unsafe { test_image(in_img) };
    // display_picture_and_wait("main() 2", &x);
}


// ============= Graphic ===============

fn display_picture_and_wait (title :&str, img : &Mat ) {
    imshow(title, img);
    let _key = wait_key(0);
}

// Canny image
fn process_img_canny(img: &Mat, dst: &mut Mat, show: bool){
    canny(&img,dst,0.,50.,3,false);
    if true == show {
        imshow("process_img_canny", dst);
        let key = wait_key(0);
    }
}

// Gray image
fn process_img_gray(img: &Mat, dst: &mut Mat,  show: bool){
    cvt_color(&img, dst, COLOR_BGR2GRAY, 0).unwrap();
    if true == show {
        imshow("process_img_gray", dst);
        let key = wait_key(0);
    }
}

// Threshold image
fn process_img_threshold(img: &Mat, dst: &mut Mat, thresh: f64, maxval: f64,  show: bool){
    threshold(&img, dst, thresh, maxval, THRESH_BINARY | THRESH_OTSU);
    if true == show {
        imshow("process_img_threshold", dst);
        let key = wait_key(0);
    }
}

// Threshold image
fn process_img_bitwise_not(img: &Mat, dst: &mut Mat,  show: bool){
    let temp = no_array().unwrap();
    bitwise_not(&img, dst, &temp);
    if true == show {
        imshow("process_img_bitwise_not", dst);
        let key = wait_key(0);
    }
}

// process_img_add_weighted
fn process_img_add_weighted(img: &Mat, dst: &mut Mat,  show: bool) {

    const COLOR: f64 = 50.0;
    let scalar = Scalar::new(COLOR, COLOR, COLOR, COLOR);
    let src2 = Mat::new_size_with_default(img.size().unwrap(), CV_8UC1, scalar).unwrap();
    add_weighted(&img, 1.5, &src2, -0.5, 1., dst, -1);
    if true == show {
        imshow("process_img_add_weighted", dst);
        let key = wait_key(0);
    }
}

// process_img_resize
fn process_img_resize(img: &Mat, dst: &mut Mat, mul: i32, show: bool) {
    resize(&img, dst, Size { width: CORNER_WIDTH * mul, height: CORNER_HEIGHT * mul }, 0., 0., INTER_LINEAR);
    if true == show {
        imshow("process_img_resize", dst);
        let key = wait_key(0);
    }
}

// process_img_morphology_ex
fn process_img_morphology_ex(img: &Mat,  dst: &mut  Mat, morph_size: i32,  morph_elem: i32, show: bool) {
    let element = get_structuring_element( morph_elem, Size{ width:2*morph_size + 1, height:  2*morph_size+1} ).unwrap();
    morphology_ex( &img, dst, 3, &element ,Point::new(-1,-1),1,BORDER_CONSTANT,morphology_default_border_value().unwrap());

    if true == show {
        display_picture_and_wait("process_img_morphology_ex", dst);
    }
}

// process_crop_by_img
fn process_crop_by_img (img: &Mat,  img_size: &Mat, show: bool) -> Mat {
    let mut roi = bounding_rect(img_size).unwrap();
    let dst = Mat::roi(&img, roi).unwrap();
    if true == show {
        display_picture_and_wait("process_crop_by_img", &dst);
    }
    dst
}

fn process_crop_img_by_size (img: &Mat,  x:i32, y:i32, width:i32, height:i32, show: bool) -> Mat {
    let roi = Rect::new(x, y, width, height);
    let dst = Mat::roi(&img, roi).unwrap();
    if true == show {
        display_picture_and_wait("process_crop_by_img", &dst);
    }
    dst
}


// get_contours
fn get_contours(img: &Mat, vec: &mut Vector<Mat>, zero_offset: Point_<i32>) {
    match find_contours(&img, vec, RETR_LIST, CHAIN_APPROX_NONE, zero_offset) {
        Ok(_ok) => {
            println!("[OK] find contours");
        }
        Err(error) => {
            println!("[KO] find contours: {}", error);
        }
    };
}

fn get_count_red(img: &Mat) -> i32 {

    // imshow("get count red", &img);
    // let _key = wait_key(0);

    let mut c_red = 0;
    for y in 0..img.rows() {
        for x in 0..img.cols() {

            let color = img.at_pt::<Vec3b>(Point::new(x, y)).unwrap().0;
            // println!("{:?}", color);
            if color[2] > color[0] && color[2] > color[1]{
                c_red += 1;
            }
        }
    }

    c_red
}

fn chk_big_card(img: &Mat) -> bool {

    let mut res = Mat::default().unwrap();
    unsafe {
      res =   auto_close_line(img.clone().unwrap());
       // println!(">>> num -> {}",num);
    };

    let mut is_big_card = false;

    let zero_offset = Point::new(0, 0);
    let mut contours_vec= Vector::new();
        get_contours(&res, &mut contours_vec, zero_offset);
        for cnt in contours_vec.iter() {
            let area = contour_area(&cnt, false).unwrap();
            if 20000. > area {
                continue;
            }
            is_big_card = true;
        }
    is_big_card
}

fn sub_image(img: &Mat, center: Point_<f32>, theta: f64, width: i32, height: i32) -> Mat {
    let shape = Size_::new(img.cols(), img.rows());
    let mut new_img = Mat::default().unwrap();
    let rot = get_rotation_matrix_2d(center, theta, 1f64);

    let bbox = RotatedRect::new(center, Size2f::new(width as f32, height as f32), theta as f32).unwrap();
    let bbox = bbox.bounding_rect2f().unwrap();
    let x1 = bbox.width / 2.0 - (img.cols() / 2) as f32;
    let x2 = bbox.height / 2.0 - (img.rows() / 2) as f32;

    match rot {
        Ok(r) => {
            r.at::<i32>(x1 as i32);
            r.at::<i32>(x2 as i32);

            warp_affine(img, new_img.borrow_mut(), &r, shape, INTER_LINEAR, BORDER_CONSTANT, Scalar::default());

        }
        _ => {}
    }

    return new_img;
}

fn find_angle(src: &Mat) -> f32 {

    // imshow("CARD  000 SRC  ", &src);
    // let _key = wait_key(0);
    let mut gray_img = src.clone().unwrap();
    let size = src.size().unwrap();
    cvt_color(&src, &mut gray_img, COLOR_BGR2GRAY, 0).unwrap();
    let src = gray_img.clone().unwrap();
    canny(&src,&mut gray_img,0.,50.,3,false);
    imshow("CARD  SRC2  ", &gray_img);
    let _key = wait_key(0);
   // const COLOR: f64 = 100.0;
   // let scalar = Scalar::new(COLOR, COLOR, COLOR, COLOR);
    //let src2 = Mat::new_size_with_default(size, CV_8UC1, scalar).unwrap();
    //let mut dst = Mat::default().unwrap();
    //compare(&gray_img, &src2, &mut dst, CMP_GT);

    //let angle =  get_angle(&gray_img);
    //TODO error check on convert
    // let _result = imshow("window_name", &dst);
    //let _key = wait_key(0);

    let zero_offset = Point::new(0, 0);
    let mut vec = VectorOfMat::new(); //findContours accepts VECTOR of Mat|UMat|Vector
    match find_contours(&mut gray_img, &mut vec, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE, zero_offset) {
        Ok(_ok) => {
            println!("[OK] find contours angle");
        }
        Err(error) => {
            println!("[KO] find contours: {}", error);
        }
    };

    let mut src = Mat::default().unwrap();
    // println!("CONTOURS INFO: empty? : {} , lenght {}", vec.is_empty(), vec.len()); // FIXME ISSUE :? :#? debug trait not implemented
    let mut _min_area_rect = RotatedRect::default().unwrap();
    let mut angle = 0.0;
    let mut large = 0.0;
    for cnt in vec.iter() {
        _min_area_rect = min_area_rect(&cnt).unwrap();
        let area = contour_area(&cnt,false).unwrap();
        println!("area {} angle {}",area, _min_area_rect.angle());
         if area > large  {
             large = area;
             angle = _min_area_rect.angle()
         }

       // let mut pts = [Point2f::new(0.0, 0.0), Point2f::new(0.0, 0.0), Point2f::new(0.0, 0.0), Point2f::new(0.0, 0.0)];
       // _min_area_rect.points(&mut pts);
        //println!(" ANGLE : {}", _min_area_rect.angle());
      /*  for j in 0..4 {
            let scalar = Scalar { 0: [0.0, 0.0, 255.0, 0.0] };
            let pt1 = Point { x: pts[j].x as i32, y: pts[j].x as i32 };
            let n = (j + 1) % 4;
            let pt2 = Point { x: pts[n].x as i32, y: pts[n].x as i32 };
            line(&mut src, pt1, pt2, scalar, 1, LINE_AA, 0);
        }*/
    }

    println!(">>> min_area_rect.angle() -> {} " , _min_area_rect.angle());
    let mut add_angle = 90.;
    if angle  < -50.  {angle  =  add_angle + angle;} else {angle = angle - 1.0};
    angle
}

fn get_angle(src: &Mat) -> f64 {
    let size = src.size().unwrap();
    let temp = no_array().unwrap();
    let mut dst = Mat::default().unwrap();
    bitwise_not(&src, &mut dst, &temp);
    //imshow("window_name", &dst);
    //wait_key(0);
    let mut lines = VectorOfVec4i::new();
    hough_lines_p(&dst, &mut lines, 1.0, (CV_PI / 180.0) as f64, 100, (size.width / 2) as f64, 20 as f64);
    const BLACK_COLOR: f64 = 0.0;
    //let scalar = Scalar::new(BLACK_COLOR, BLACK_COLOR, BLACK_COLOR, BLACK_COLOR);
    //let mut disp_lines = Mat::new_size_with_default(size, CV_8UC1, scalar).unwrap();
    let mut angle = 0.;


    let nb_lines = lines.len();
    for l in lines {
        //let point = Point::new(l[0], l[1]);
        /*   line(&mut disp_lines, point,
             Point::new(l[2], l[3]), Scalar::new(255., 0., 0., 0.), 0, 0, 0);*/
        let y = (l[3] - l[1]) as f64;
        let x = (l[2] - l[0]) as f64;
        angle += atan2(y, x);
    }
    angle /= nb_lines as f64;
    let rs = angle * 180.0 / CV_PI as f64;
    println!("Angle  {} : ", rs);
    rs
}
