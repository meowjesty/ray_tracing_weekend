use std::{
    fs::{File, OpenOptions},
    io::Write,
};

use glam::Vec3;

type Point3 = glam::Vec3;
type Color = glam::Vec3;

#[derive(Debug)]
struct Ray {
    pub origin: Point3,
    pub direction: glam::Vec3,
}

impl Ray {
    // P(t) = O + tB
    pub fn at(&self, t: f32) -> Point3 {
        self.origin + t * self.direction
    }

    /// Linearly blends white and blue depending on the height of the `y` coordinate _after_
    /// scaling the ray direction to unit length (`-1.0 < y < 1.0`).
    ///
    /// - When `t = 1.0` -> blue (end value);
    /// - When `t = 0.0` -> white (start value);
    /// - In between -> blend (linear interpolation);
    ///
    /// `blended_value = (1 - t) * start_value + t * end_value`
    pub fn color(&self) -> Color {
        let unit = self.direction.normalize();
        let t = 0.5 * (unit.y() + 1.0);
        let start_value = Color::new(1.0, 1.0, 1.0); // white
        let end_value = Color::new(0.5, 0.7, 1.0); // blue
        let blended_value = (1.0 - t) * start_value + t * end_value;
        // let blended_value = start_value.lerp(end_value, t);
        blended_value
    }
}

fn write_color(pixel_color: Color) -> String {
    let r: u32 = (255.999 * pixel_color.x()) as u32;
    let g: u32 = (255.999 * pixel_color.y()) as u32;
    let b: u32 = (255.999 * pixel_color.z()) as u32;

    format!("{} {} {}\n", r, g, b)
}

fn listing_9() -> std::io::Result<()> {
    println!("Listing 9");
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open("./images/listing-9.ppm")?;
    let mut file_contents = String::with_capacity(65_536);

    // Image
    let aspect_ratio = 16.0 / 9.0;
    let image_width = 400;
    let image_height = (image_width as f32 / aspect_ratio) as u32;

    // Camera
    let viewport_height = 2.0;
    let viewport_width = aspect_ratio * viewport_height;
    let focal_length = 1.0;
    let origin = Point3::zero();
    let horizontal = Vec3::new(viewport_width, 0.0, 0.0);
    let vertical = Vec3::new(0.0, viewport_height, 0.0);
    let lower_left_corner =
        origin - horizontal / 2.0 - vertical / 2.0 - Vec3::new(0.0, 0.0, focal_length);

    // Render
    file_contents.push_str(&format!("P3\n{} {}\n255\n", image_width, image_height));

    for j in (0..image_height).rev() {
        println!("Scanlines remaining: {}", j);
        for i in 0..image_width {
            let u = i as f32 / (image_width - 1) as f32;
            let v = j as f32 / (image_height - 1) as f32;
            let ray = Ray {
                origin,
                direction: lower_left_corner + u * horizontal + v * vertical - origin,
            };
            let pixel_color = ray.color();
            file_contents.push_str(&write_color(pixel_color));
        }
    }

    file.write_all(file_contents.as_bytes())?;
    println!("Done!");

    Ok(())
}

fn listing_1() -> std::io::Result<()> {
    println!("Listing 1");
    let mut file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open("./images/listing-1.ppm")?;
    let image_width = 256;
    let image_height = 256;
    let mut file_contents = String::with_capacity(65_536);

    file_contents.push_str(&format!("P3\n{} {}\n255\n", image_width, image_height));

    for j in (0..image_height).rev() {
        println!("Scanlines remaining: {}", j);
        for i in 0..image_width {
            let pixel_color = Color::new(
                i as f32 / (image_width - 1) as f32,
                j as f32 / (image_height - 1) as f32,
                0.25,
            );
            file_contents.push_str(&write_color(pixel_color));
        }
    }

    file.write_all(file_contents.as_bytes())?;
    println!("Done!");

    Ok(())
}

fn main() {
    let _app = listing_1();
    let _app = listing_9();
}
