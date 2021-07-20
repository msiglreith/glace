use glace::{compute::reorder256_xyyxxyyx, u32x2};

fn main() {
    let mut grid = [[0; 16]; 16];
    for i in 0..256 {
        let u32x2 { x, y } = reorder256_xyyxxyyx(i);
        grid[y as usize][x as usize] = i;
    }

    for i in 0..16 {
        println!("{:>3?}", grid[i]);
    }
}
