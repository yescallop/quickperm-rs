use quickperm::Perm;

fn main() {
    let mut arr = [1, 2, 3];
    arr.permute(|perm| println!("{:?}", perm));
}
