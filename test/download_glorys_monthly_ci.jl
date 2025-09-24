dir = "./GLORYS_data"
mkpath(dir)
zipped_filepath = joinpath(dir, "glorys_data.zip")

download("https://figshare.com/ndownloader/articles/30193228/versions/1", zipped_filepath)

run(`unzip -o $zipped_filepath -d $dir`)

@info "GLORYS data downloaded and extracted to $dir"