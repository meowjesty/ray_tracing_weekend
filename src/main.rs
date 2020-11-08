#![feature(clamp)]

use std::{
    fs::{File, OpenOptions},
    io::{BufWriter, Write},
    rc::Rc,
    sync::Arc,
};

use rand::Rng;

use glam::Vec3;

#[derive(Debug, Clone)]
pub struct Camera {
    aspect_ratio: f32,
    viewport_height: f32,
    viewport_width: f32,
    focal_length: f32,
    origin: Point3,
    lower_left_corner: Point3,
    horizontal: Vec3,
    vertical: Vec3,
}

impl Camera {
    pub fn get_ray(&self, u: f32, v: f32) -> Ray {
        Ray {
            origin: self.origin,
            direction: self.lower_left_corner + u * self.horizontal + v * self.vertical
                - self.origin,
        }
    }
}

impl Default for Camera {
    fn default() -> Self {
        let aspect_ratio = 16.0 / 9.0;
        let viewport_height = 2.0;
        let viewport_width = aspect_ratio * viewport_height;
        let focal_length = 1.0;
        let origin = Point3::default();
        let horizontal = Vec3::new(viewport_width, 0.0, 0.0);
        let vertical = Vec3::new(0.0, viewport_height, 0.0);
        let lower_left_corner =
            origin - horizontal / 2.0 - vertical / 2.0 - Vec3::new(0.0, 0.0, focal_length);

        Self {
            aspect_ratio,
            viewport_height,
            viewport_width,
            focal_length,
            origin,
            lower_left_corner,
            horizontal,
            vertical,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct HitRecord {
    point: Point3,
    normal: Vec3,
    t: f32,
    /// The ray may be coming from outside or inside, and there are 2 ways of setting up the normal
    // to handle it:
    ///
    /// 1. The normal always points from center to intersection;
    ///     - if the ray intersects from the outside, the normal will point against it;
    ///     - if the ray intersects from the inside, the normal will point with it;
    ///     - to calculate the color, just check if the normal is in favor or against the ray;
    ///     - `if ray.direction.dot(outward_normal) > 0.0 then "ray inside" else "ray outside"`;
    /// 2. The normal always points against the ray:
    ///     - if the ray intersects from the outside, the normal will point against it;
    ///     - if the ray intersects from the inside, the normal will point against it;
    ///     - to calculate the color, just check if the normal is in favor or against the ray;
    ///     - cannot use the dot product to determine which side of the surface the ray is on,
    ///     must store it;
    ///
    /// We're using `2` here.
    front_face: bool,
}

impl HitRecord {
    pub fn face_normal(ray: &Ray, outward_normal: Vec3) -> (bool, Vec3) {
        let front_face = ray.direction.dot(outward_normal) < 0.0;
        if front_face {
            (front_face, outward_normal)
        } else {
            (front_face, -outward_normal)
        }
    }
}

/// This trait helps when dealing with multiple spheres (or whatever objects).
pub trait Hittable {
    /// The hit only "counts" if `tₘᵢₙ < t < tₘₐₓ`.
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord>;
}

pub struct HittableList {
    list: Vec<Rc<dyn Hittable>>,
}

impl Hittable for HittableList {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        let mut hit = None;
        let mut closest_so_far = t_max;
        for obj in self.list.iter() {
            let hit_something = obj.hit(ray, t_min, closest_so_far);
            match hit_something {
                Some(ref record) => {
                    hit = Some(record.clone());
                    closest_so_far = record.t;
                }
                None => (),
            }
        }

        hit
    }
}

#[derive(Debug)]
pub struct Sphere {
    center: Point3,
    radius: f32,
}

impl Hittable for Sphere {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<HitRecord> {
        let oc = ray.origin - self.center;
        let a = ray.direction.length_squared();
        let half_b = oc.dot(ray.direction);
        let c = oc.length_squared() - self.radius * self.radius;
        let discriminant = half_b * half_b - a * c;
        if discriminant < 0.0 {
            return None;
        }
        let sqrtd = discriminant.sqrt();

        // NOTE(alex): Find the nearest root that lies in the acceptable range.
        let mut root = (-half_b - sqrtd) / a;
        if root < t_min || root > t_max {
            root = (-half_b + sqrtd) / a;

            if root < t_min || root > t_max {
                return None;
            }
        }

        let t = root;
        let point = ray.at(t);
        let outward_normal = (point - self.center) / self.radius;
        let (front_face, normal) = HitRecord::face_normal(ray, outward_normal);
        let hit_record = HitRecord {
            t,
            point,
            normal,
            front_face,
        };
        Some(hit_record)
    }
}

type Point3 = glam::Vec3;
type Color = glam::Vec3;

#[derive(Debug)]
pub struct Ray {
    pub origin: Point3,
    pub direction: glam::Vec3,
}

impl Ray {
    /// `P(t) = A + (t * b)`
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
    ///
    /// `depth` prevents possible stack overflow due to recursive nature.
    pub fn color(&self, world: &dyn Hittable, depth: u32) -> Color {
        // NOTE(alex): Too many recursions, the ray bounced too much, no more light is gathered.
        if depth == 0 {
            return Color::new(0.0, 0.0, 0.0);
        }

        // NOTE(alex): `0.001` here fixes the shadow _acne_ problem, when the ray hits and is
        // reflecting at `t = -0.0000001` or `t = 0.0000001` (floating point approximations), so
        // we ingore hits very near zero.
        if let Some(hit) = world.hit(self, 0.001, f32::INFINITY) {
            // let target: Point3 = hit.point + hit.normal + Vec3::random_unit_vector();
            let target: Point3 = hit.point + Vec3::random_in_hemisphere(hit.normal);
            // NOTE(alex): This ray represents the random point we've selected inside the
            // hit sphere, so `origin` is a random point and `direction` is a vector from
            // the hit surface point to this random point.
            let random_ray = Ray {
                origin: hit.point,
                direction: target - hit.point,
            };
            0.5 * random_ray.color(world, depth - 1)
        } else {
            let unit_direction = self.direction.normalize();
            let t = 0.5 * (unit_direction.y() + 1.0);
            let start_value = Color::new(1.0, 1.0, 1.0); // white
            let end_value = Color::new(0.5, 0.7, 1.0); // blue
            let blended_value = (1.0 - t) * start_value + t * end_value;
            // let blended_value = start_value.lerp(end_value, t);
            blended_value
        }
    }

    /// Ray-Sphere Intersection:
    ///
    /// For a sphere with radius `R` centered at the origin:
    ///
    /// - A point `(x, y, z)` is **ON** the sphere if `(x² + y² + z²) = R²`;
    /// - `(x, y, z)` is **inside** the sphere if `(x² + y² + z²) < R²`;
    /// - `(x, y, z)` is **outside** the sphere if `(x² + y² + z²) > R²`;
    ///
    /// If the sphere center is at `(Cₓ, Cᵧ, C𝓏)` then the equation becomes:
    ///
    /// - `(x - Cₓ)² + (y - Cᵧ)² + (z - C𝓏)² = r²`;
    ///
    /// A vector from center `C = (Cₓ, Cᵧ, C𝓏)` to point `P = (x, y, z)` is
    /// `(P - C)` and therefore:
    ///
    /// - `(P - C) ⋅ (P - C) = (x - Cₓ)² + (y - Cᵧ)² + (z - C𝓏)²`;
    /// - Shorthand `(P - C) ⋅ (P - C) = r²`;
    ///
    /// Substituting with `P(t) = A + (t * b) we get a quadratic equation that:
    ///
    /// - `0` roots means no intersection (ray is outside);
    /// - `1` root means one intersection (ray touches once);
    /// - `2` roots means two intersections (ray touches twice);
    ///
    /// Refer to sphere_roots.jpg for a visualization.
    pub fn hit_sphere(&self, center: Point3, radius: f32) -> f32 {
        let origin_to_center = self.origin - center;
        // NOTE(alex): A vector dotted with itself is equal to the squared length of that vector.
        // V ⋅ V = V.length²
        let a = self.direction.length_squared();
        // NOTE(alex): If `b = 2h`, the equation becomes:
        // `(-2h +- sqrt((2h)² - 4ac)) / 2a` ->
        // `(-2h +- 2 * sqrt(h² - ac)) / 2a` ->
        // `(-h +- sqrt(h² - ac)) / a`
        let half_b = origin_to_center.dot(self.direction);
        let c = origin_to_center.length_squared() - radius * radius;
        // NOTE(alex): Solving the quadratic equation to determine if there is an intersection,
        // and how many.
        // The discriminant is the `sqrt(b² -4ac)` part of the equation, here in simplified form
        // thanks to the `2h` idea.
        let discriminant = half_b * half_b - a * c;

        if discriminant < 0.0 {
            -1.0
        } else {
            (-half_b - discriminant.sqrt()) / a
        }
    }
}

pub trait Vec3Random {
    fn random() -> Vec3;
    fn random_bounded(min: f32, max: f32) -> Vec3;
    /// This is used in a rejection selection.
    fn random_in_unit_sphere() -> Vec3;
    /// True lambertian Reflection:
    ///
    /// Pick random points on the surface of the unit sphere, offset along the surface normal.
    /// Picking random points can be achieved by picking random points **in** the unit sphere,
    /// and then normalizing those.
    ///
    /// See the `generating_a_random_unit_vector` picture for reference.
    fn random_unit_vector() -> Vec3;
    /// Uniform scatter direction for all angles away from the hit point, with no dependence on
    /// the angle from the normal. (Used in raytracers before the adoption of Lambertian diffuse)
    fn random_in_hemisphere(normal: Vec3) -> Vec3;
}

impl Vec3Random for Vec3 {
    fn random() -> Vec3 {
        let mut rng = rand::thread_rng();
        let x = rng.gen_range(-1.0..=1.0);
        let y = rng.gen_range(-1.0..=1.0);
        let z = rng.gen_range(-1.0..=1.0);
        Vec3::new(x, y, z)
    }

    fn random_bounded(min: f32, max: f32) -> Vec3 {
        let mut rng = rand::thread_rng();
        let x = rng.gen_range(min..=max);
        let y = rng.gen_range(min..=max);
        let z = rng.gen_range(min..=max);
        Vec3::new(x, y, z)
    }

    fn random_in_unit_sphere() -> Vec3 {
        loop {
            let point = Vec3::random_bounded(-1.0, 1.0);

            if point.length_squared() < 1.0 {
                return point;
            }
        }
    }

    fn random_unit_vector() -> Vec3 {
        Vec3::random_in_unit_sphere().normalize()
    }

    fn random_in_hemisphere(normal: Vec3) -> Vec3 {
        let in_unit_sphere = Vec3::random_in_unit_sphere();

        // NOTE(alex): In the same hemisphere as the normal
        if in_unit_sphere.dot(normal) > 0.0 {
            in_unit_sphere
        } else {
            -in_unit_sphere
        }
    }
}

fn write_color(pixel_color: Color, samples_per_pixel: u32) -> String {
    let r = pixel_color.x();
    let g = pixel_color.y();
    let b = pixel_color.z();

    // NOTE(alex): Divide the color by the number of samples. (Antialiasing)
    let scale = 1.0 / samples_per_pixel as f32;
    let r = r * scale;
    let g = g * scale;
    let b = b * scale;

    let r: u32 = (256.0 * r.clamp(0.0, 0.999)) as u32;
    let g: u32 = (256.0 * g.clamp(0.0, 0.999)) as u32;
    let b: u32 = (256.0 * b.clamp(0.0, 0.999)) as u32;

    format!("{} {} {}\n", r, g, b)
}

fn get_color(pixel_color: Color, samples_per_pixel: u32) -> Vec<u8> {
    let r = pixel_color.x();
    let g = pixel_color.y();
    let b = pixel_color.z();

    // NOTE(alex): Divide the color by the number of samples. (Antialiasing)
    // NOTE(alex): Square root for gamma-correction (for gamma = 2.0).
    let scale = 1.0 / samples_per_pixel as f32;
    let r = (r * scale).sqrt();
    let g = (g * scale).sqrt();
    let b = (b * scale).sqrt();

    let r: u8 = (256.0 * r.clamp(0.0, 0.999)) as u8;
    let g: u8 = (256.0 * g.clamp(0.0, 0.999)) as u8;
    let b: u8 = (256.0 * b.clamp(0.0, 0.999)) as u8;

    vec![r, g, b]
}

/// Hack for rays hitting a diffuse (matte) surface:
/// There are 2 unit radius spheres tangent to the hit point `P` of a surface. They have
/// a center point `(P + n)` and `(P - n)`, where `n` is the normal of the surface.
/// - Sphere with center `(P + n)` is outside the surface;
/// - Sphere with center `(P - n)` is inside the surface;
///
/// Select the tangent unit radius sphere that is on the same side of the surface
/// as the ray origin. Pick a random point `S` inside this sphere and send a ray from the hit
/// point `P` to the random point `S`: `(S - P)` vector.

fn app() -> std::io::Result<()> {
    println!("Open file and generate image!");
    let filename = "8.25";
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        // .open("./images/listing-30.ppm")?;
        .open(format!("./images/{}.png", filename))?;
    // let decoder = png::Decoder::new(file);
    let buf_writer = BufWriter::new(file);

    // Image
    let aspect_ratio = 16.0 / 9.0;
    let image_width = 400;
    let image_height = (image_width as f32 / aspect_ratio) as u32;
    let samples_per_pixel = 100;
    let mut encoder = png::Encoder::new(buf_writer, image_width, image_height);
    encoder.set_color(png::ColorType::RGB);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;
    let mut image_buffer = Vec::with_capacity(65_653);
    let max_depth = 50;

    // World
    let world = HittableList {
        list: vec![
            Rc::new(Sphere {
                center: Point3::new(0.0, 0.0, -1.0),
                radius: 0.5,
            }),
            Rc::new(Sphere {
                center: Point3::new(0.0, -100.5, -1.0),
                radius: 100.0,
            }),
        ],
    };

    // Camera
    let camera = Camera::default();

    // Random
    let mut rng = rand::thread_rng();

    // Render
    println!("Scanline running ...");
    for j in (0..image_height).rev() {
        println!("Scanlines remaining {}.", j);
        for i in 0..image_width {
            let mut pixel_color = Color::default();

            for s in 0..samples_per_pixel {
                let random_x: f32 = rng.gen();
                let random_y: f32 = rng.gen();

                let u = (i as f32 + random_x) / (image_width - 1) as f32;
                let v = (j as f32 + random_y) / (image_height - 1) as f32;

                let ray = camera.get_ray(u, v);
                pixel_color += ray.color(&world, max_depth);
            }
            image_buffer.push(get_color(pixel_color, samples_per_pixel));
        }
    }
    println!("Scanline finished!");

    let flat: Vec<u8> = image_buffer.into_iter().flatten().collect();
    println!("Writing to file.");
    writer.write_image_data(&flat)?;
    println!("Done!");

    Ok(())
}

fn main() {
    // let _app = listing_1();
    // let _app = listing_9();
    let _app = app();
}
