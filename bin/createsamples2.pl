#!/usr/bin/perl
use File::Basename;
use strict;


##########################################################################
# Create samples from an image applying distortions repeatedly
# (create many many samples from many images applying distortions)
#
#  perl createtrainsamples.pl <positives.dat> <negatives.dat> <vec_output_dir>
#      [<totalnum = 7000>] [<createsample_command_options = ./createsamples -w 20 -h 20...>]
#  ex) perl createtrainsamples.pl positives.dat negatives.dat samples
#
# Author: Naotoshi Seo
# Date  : 09/12/2008 Add <totalnum> and <createsample_command_options> options
# Date  : 06/02/2007
# Date  : 03/12/2006
#########################################################################
my $cmd = './createsamples -bgcolor 0 -bgthresh 0 -maxxangle 1.1 -maxyangle 1.1 maxzangle 0.5 -maxidev 40 -w 20 -h 20';
my $totalnum = 7000;
my $tmpfile  = 'tmp';

if ($#ARGV < 2) {
    print "Usage: perl createtrainsamples.pl\n";
    print "  <positives_collection_filename>\n";
    print "  <negatives_collection_filename>\n";
    print "  <output_dirname>\n";
    print "  [<totalnum = " . $totalnum . ">]\n";
    print "  [<createsample_command_options = '" . $cmd . "'>]\n";
    exit;
}
my $positive  = $ARGV[0];
my $negative  = $ARGV[1];
my $outputdir = $ARGV[2];
$totalnum     = $ARGV[3] if ($#ARGV > 2);
$cmd          = $ARGV[4] if ($#ARGV > 3);


print "arg0:".$ARGV[0]."\n";
#print "arg1:".$ARGV[1];
#print "arg2:".$ARGV[2];
#print "arg3:".$ARGV[3];
#print "arg4:".$ARGV[4];


my $tool_name = "opencv_createsamples";  # simple example
my $tool_path = '';

for my $path ( split /:/, $ENV{PATH} ) {
   if ( -f "$path/$tool_name" && -x _ ) {
       ##print "$tool_name found in $path\n";
       $tool_path = "$path/$tool_name";
       last;
   }
}
die "404" unless ( $tool_path );

open(POSITIVE, "< $positive");
my @positives = <POSITIVE>;
close(POSITIVE);

open(NEGATIVE, "< $negative");
my @negatives = <NEGATIVE>;
close(NEGATIVE);


# number of generated images from one image so that total will be $totalnum
print "postives: ".$#positives."\n";
print "totalnum: ".$totalnum."\n";
my $numfloor  = int($totalnum / $#positives);
my $numremain = $totalnum - $numfloor * $#positives;

# Get the directory name of positives
my $first = $positives[0];
my $last  = $positives[$#positives];
while ($first ne $last) {
    $first = dirname($first);
    $last  = dirname($last);
    if ( $first eq "" ) { last; }
}
my $imgdir = $first;
my $imgdirlen = length($first);



for (my $k = 0; $k < $#positives; $k++ ) {
    my $img = $positives[$k];
    my $num = ($k < $numremain) ? $numfloor + 1 : $numfloor;

    # Pick up negative images randomly
    my @localnegatives = ();
    for (my $i = 0; $i < $num; $i++) {
        my $ind = int(rand($#negatives));
        push(@localnegatives, $negatives[$ind]);
    }
    open(TMP, "> $tmpfile");
    print TMP @localnegatives;
    close(TMP);
    #system("cat $tmpfile");

    !chomp($img);
    my $vec = $outputdir . substr($img, $imgdirlen) . ".vec" ;
    print "$cmd -img $img -bg $tmpfile -vec $vec -num $num" . "\n";
    system("$cmd -img $img -bg $tmpfile -vec $vec -num $num");
}
unlink($tmpfile);