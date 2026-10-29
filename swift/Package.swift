// swift-tools-version:6.0
// Sortana for Swift: the sortana-ffi static library plus its generated
// UniFFI bindings. Built by scripts/build-swift.sh (macOS).
import PackageDescription

let package = Package(
  name: "Sortana",
  platforms: [.iOS(.v18), .macOS(.v14)],
  products: [
    .library(name: "Sortana", targets: ["Sortana"]),
  ],
  targets: [
    .binaryTarget(name: "SortanaFFI", path: "SortanaFFI.xcframework"),
    .target(name: "Sortana", dependencies: ["SortanaFFI"], path: "Sources/Sortana"),
  ]
)
