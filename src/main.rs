#![feature(clamp)]

use std::{
    f32::consts::PI,
    fs::{File, OpenOptions},
    io::{BufWriter, Write},
    rc::Rc,
    sync::Arc,
};

use rand::Rng;
use rayon::prelude::*;

use glam::Vec3;

mod vec3ext;
use vec3ext::*;

/// Dieletrics are clear materials (water, glass, diamonds, ...), when a light ray hits them
/// it splits into a reflected ray and a refracted (transmitted) ray.

#[derive(Debug, Clone)]
pub struct Camera {
    origin: Point3,
    lower_left_corner: Point3,
    horizontal: Vec3,
    vertical: Vec3,
}

impl Camera {
    pub fn new(vertical_fov: f32, aspect_ratio: f32) -> Self {
        let theta = vertical_fov.to_radians();
        let h = (theta / 2.0).tan();
        let viewport_height = 2.0 * h;
        let viewport_width = aspect_ratio * viewport_height;

        let focal_length = 1.0;
        let origin = Point3::new(0.0, 0.0, 0.0);
        let horizontal = Vec3::new(viewport_width, 0.0, 0.0);
        let vertical = Vec3::new(0.0, viewport_height, 0.0);
        let lower_left_corner =
            origin - horizontal / 2.0 - vertical / 2.0 - Vec3::new(0.0, 0.0, focal_length);

        Self {
            origin,
            horizontal,
            vertical,
            lower_left_corner,
        }
    }
    pub fn get_ray(&self, u: f32, v: f32) -> Ray {
        Ray {
            origin: self.origin,
            direction: self.lower_left_corner + u * self.horizontal + v * self.vertical
                - self.origin,
        }
    }
}

#[derive(Clone)]
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
pub trait Hittable: Send + Sync {
    /// The hit only "counts" if `tₘᵢₙ < t < tₘₐₓ`.
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<(HitRecord, Arc<dyn Material>)>;
}

pub struct HittableList {
    list: Vec<Arc<dyn Hittable>>,
}

pub trait Material {
    fn scatter(&self, ray_in: &Ray, hit: &HitRecord) -> Option<(Ray, Color)>;
}

#[derive(Debug, Clone)]
pub struct LambertianMaterial {
    albedo: Color,
}

impl Material for LambertianMaterial {
    fn scatter(&self, ray_in: &Ray, hit: &HitRecord) -> Option<(Ray, Color)> {
        let mut scatter_direction = hit.normal + Vec3::random_unit_vector();

        // Catch degenerate scatter direction.
        if scatter_direction.near_zero() {
            scatter_direction = hit.normal;
        }

        let scattered = Ray {
            origin: hit.point,
            direction: scatter_direction,
        };
        Some((scattered, self.albedo))
    }
}

#[derive(Debug, Clone)]
pub struct MetalMaterial {
    albedo: Color,
    /// Randomize the reflected direction by using a small sphere and choosing a new endpoint for
    /// the ray. Refer to `generating_fuzzed_reflection_rays` for a visualization.
    /// The fuzziness is the radius of this sphere, and if it scatters bellow the hit surface, then
    /// it counts as absorbed.
    fuzz: f32,
}

impl Material for MetalMaterial {
    fn scatter(&self, ray_in: &Ray, hit: &HitRecord) -> Option<(Ray, Color)> {
        let reflected = ray_in.direction.reflect(hit.normal);
        let scattered = Ray {
            origin: hit.point,
            direction: reflected + self.fuzz * Vec3::random_in_unit_sphere(),
        };
        if scattered.direction.dot(hit.normal) > 0.0 {
            Some((scattered, self.albedo))
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
pub struct DialetricMaterial {
    index_of_refraction: f32,
}

impl DialetricMaterial {
    /// Schlick's approximation for reflectance.
    pub fn reflectance(cosine: f32, ref_index: f32) -> f32 {
        let mut r0 = (1.0 - ref_index) / (1.0 + ref_index);
        r0 = r0 * r0;
        r0 + (1.0 - r0) * (1.0 - cosine).powi(5)
    }
}

impl Material for DialetricMaterial {
    fn scatter(&self, ray_in: &Ray, hit: &HitRecord) -> Option<(Ray, Color)> {
        // NOTE(alex): The glass surface absorbs nothing.
        let attenuation = Color::new(1.0, 1.0, 1.0);
        let refraction_ratio = if hit.front_face {
            1.0 / self.index_of_refraction
        } else {
            self.index_of_refraction
        };

        let unit_direction = ray_in.direction.normalize();

        // NOTE(alex): When the ray is in the material with higher refractive index, there is no
        // real solution to Snell's law, this means no refraction possible.
        // Consider the case where the ray is inside glass (`n = 1.5`) and
        // outside is air (`n = 1.0`):
        // `sinθ = (1.5 / 1.0) * sinθ`, `sinθ` cannot be greater than `1`, so if:
        // `(1.5 / 1.0) * sinθ > 1.0` the equality is broken and no solution exists (no refraction).
        let cos_theta = (-unit_direction).dot(hit.normal).min(1.0);
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
        let cannot_refract = refraction_ratio * sin_theta > 1.0;
        let mut rng = rand::thread_rng();
        let direction = if cannot_refract
            || DialetricMaterial::reflectance(cos_theta, refraction_ratio) > rng.gen()
        {
            // Must reflect
            unit_direction.reflect(hit.normal)
        } else {
            // Can refract
            unit_direction.refract(hit.normal, refraction_ratio)
        };

        let scattered = Ray {
            origin: hit.point,
            direction,
        };

        Some((scattered, attenuation))
    }
}

impl Hittable for HittableList {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<(HitRecord, Arc<dyn Material>)> {
        let mut hit = None;
        let mut closest_so_far = t_max;
        let mut mat = None;
        for obj in self.list.iter() {
            let hit_something = obj.hit(ray, t_min, closest_so_far);
            match hit_something {
                Some((ref record, material)) => {
                    hit = Some(record.clone());
                    closest_so_far = record.t;
                    mat = Some(material)
                }
                None => (),
            }
        }

        if hit.is_some() && mat.is_some() {
            Some((hit.unwrap(), mat.unwrap()))
        } else {
            None
        }
    }
}

#[derive(Clone)]
pub struct Sphere {
    center: Point3,
    radius: f32,
    /// Tells how the rays interact with the surface.
    material: Arc<dyn Material>,
}

unsafe impl Send for Sphere {}
unsafe impl Sync for Sphere {}

impl Hittable for Sphere {
    fn hit(&self, ray: &Ray, t_min: f32, t_max: f32) -> Option<(HitRecord, Arc<dyn Material>)> {
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
        Some((hit_record, self.material.clone()))
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
        if let Some((hit, material)) = world.hit(self, 0.001, f32::INFINITY) {
            if let Some((scattered, attenuation)) = material.scatter(self, &hit) {
                attenuation * scattered.color(world, depth - 1)
            } else {
                Color::new(0.0, 0.0, 0.0)
            }
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
    let filename = "Wide-angle view";
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
    // let image_width = 1024;
    let image_height = (image_width as f32 / aspect_ratio) as u32;
    let samples_per_pixel = 100;
    let mut encoder = png::Encoder::new(buf_writer, image_width, image_height);
    encoder.set_color(png::ColorType::RGB);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header()?;
    let mut image_buffer: Vec<Vec<u8>> = Vec::with_capacity(65_653);
    let max_depth = 50;

    // World
    let r = (PI / 4.0).cos();
    let material_ground = Arc::new(LambertianMaterial {
        albedo: Color::new(0.8, 0.8, 0.0),
    });
    let material_center = Arc::new(LambertianMaterial {
        albedo: Color::new(0.1, 0.2, 0.5),
    });
    let material_left = Arc::new(DialetricMaterial {
        index_of_refraction: 1.5,
    });
    let material_right = Arc::new(MetalMaterial {
        albedo: Color::new(0.8, 0.6, 0.2),
        fuzz: 0.0,
    });
    let world = HittableList {
        list: vec![
            Arc::new(Sphere {
                center: Point3::new(-r, 0.0, -1.0),
                radius: r,
                material: Arc::new(LambertianMaterial {
                    albedo: Color::new(0.0, 0.0, 1.0),
                }),
            }),
            Arc::new(Sphere {
                center: Point3::new(r, 0.0, -1.0),
                radius: r,
                material: Arc::new(LambertianMaterial {
                    albedo: Color::new(1.0, 0.0, 0.0),
                }),
            }),
        ],
    };
    // let world = HittableList {
    //     list: vec![
    //         Arc::new(Sphere {
    //             center: Point3::new(0.0, -100.5, -1.0),
    //             radius: 100.0,
    //             material: material_ground,
    //         }),
    //         Arc::new(Sphere {
    //             center: Point3::new(0.0, 0.0, -1.0),
    //             radius: 0.5,
    //             material: material_center,
    //         }),
    //         Arc::new(Sphere {
    //             center: Point3::new(-1.0, 0.0, -1.0),
    //             radius: 0.5,
    //             material: material_left.clone(),
    //         }),
    //         Arc::new(Sphere {
    //             center: Point3::new(-1.0, 0.0, -1.0),
    //             radius: -0.4,
    //             material: material_left,
    //         }),
    //         Arc::new(Sphere {
    //             center: Point3::new(1.0, 0.0, -1.0),
    //             radius: 0.5,
    //             material: material_right,
    //         }),
    //     ],
    // };

    // Camera
    let camera = Camera::new(90.0, aspect_ratio);

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
