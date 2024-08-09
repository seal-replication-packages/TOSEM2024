#!/usr/bin/env bash

# Usage:
#   git log2json                # for the basic commit info
#   git log2json --name-only    # for the same plus changed file list
#
# Based on https://gist.github.com/textarcana/1306223

for ((i = 1; i <= 20; i++)); do
	echo $i
	for dir in /home/19sn17/home/PyRef/outputs_2/repos/$i/*/ ; do
		project=${dir%/}
		# Check if the folder name is not "project.und"
        if [[ $(basename "$project") != "project.und" ]]; then
			echo $project
			cd $project
			git log \
				--pretty=format:'%n{%n  "commit": "%H",%n  "author": "%an <%ae>",%n  "date": "%ad",%n  "message": "%f":FILES:' \
				$@ | \
				perl -ne '
				BEGIN{print "["};
				if ($i = (/:FILES:[\n\r]*$/../^$/)) {
					if ($i == 1) {
						s/:FILES:[\n\r]*$//;
						$message = $_;
					} elsif ($i =~ /E0$/) {
						#$print_files->();
						print_files();
						@files = ();
					} elsif ($_ !~ /^$/) {
						chomp $_;
						push @files, $_;
					}
				} else { print; }
				END { print_files(1); }
				
				sub print_files {
					$last_line = shift;
					print $message; 
					@files ? 
						print qq(,\n  "files": [\n@{[join qq(,\n), map {qq(    "$_")} map {json_escape($_)} @files]}\n  ]\n}) 
						: print "\n}";
					$last_line ? print "]" : print @files ? "," : ",\n";
				};
				sub json_escape { $_ = shift; s/([\\"])/\\\1/g; return $_; }' > $project.json
			mv $project.json /home/19sn17/home/PyRef/outputs_2/commit-logs
		fi
	done
done
