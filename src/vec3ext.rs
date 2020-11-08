use glam::Vec3;
use rand::Rng;

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

pub trait Vec3NearZero {
    /// `hit.normal + Vec3::random_unit_vector()` sum may be zero if the random unit vector is
    /// exactly opposite of the normal. This leads to bad numbers, so we intercept this condition.
    fn near_zero(&self) -> bool;
}

pub trait Vec3Reflect {
    fn reflect(self, normal: Vec3) -> Vec3;
}

impl Vec3Reflect for Vec3 {
    fn reflect(self, normal: Vec3) -> Vec3 {
        self - 2.0 * self.dot(normal) * normal
    }
}

impl Vec3NearZero for Vec3 {
    fn near_zero(&self) -> bool {
        // Return true if the vector is close to zero in all dimensions.
        let s = 1e-8;
        self.x().abs() < s && self.y().abs() < s && self.z().abs() < s
    }
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
