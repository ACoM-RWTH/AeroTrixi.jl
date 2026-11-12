using Documenter
using DocumenterInterLinks
using AeroTrixi

# Set up for local builds
if (get(ENV, "CI", nothing) != "true")
    push!(LOAD_PATH, dirname(@__DIR__))
end

# Define module-wide setups such that the respective modules are available in doctests
DocMeta.setdocmeta!(AeroTrixi, :DocTestSetup, :(using AeroTrixi); recursive = true)

# Links to the documentation of Trixi.jl, e.g., [`P4estMesh`](@extref Trixi.P4estMesh)
links = InterLinks("Trixi" => "https://trixi-framework.org/TrixiDocumentation/stable/")

# Make documentation
makedocs(modules = [AeroTrixi],
         plugins = [links],
         sitename = "AeroTrixi.jl",
         format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true",
                                  canonical = "https://acom-rwth.github.io/AeroTrixi.jl/stable"),
         pages = [
             "Home" => "index.md",
             "Trixi.jl-Gridap.jl bridge" => "trixi_gridap_bridge.md",
             "API Reference" => [
                 "Overview" => "api.md",
                 "Surface quantities" => "api/surface_quantities.md",
                 "Thermodynamics" => "api/thermodynamics.md",
                 "Multi-species equations" => "api/multispecies_equations.md",
                 "Trixi.jl-Gridap.jl bridge" => "api/trixi_gridap_bridge.md"
             ]
         ],
         doctest = true,
         linkcheck = false,
         warnonly = [:missing_docs, :cross_references])

# Deploy documentation
deploydocs(repo = "github.com/ACoM-RWTH/AeroTrixi.jl.git",
           devbranch = "main",
           devurl = "dev",
           versions = ["stable" => "v^", "dev" => "dev"],
           push_preview = true)
