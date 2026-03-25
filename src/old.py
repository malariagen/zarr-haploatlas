import allel
import collections
import lzma
import pickle
import vcfpy

import numpy as np
import pandas as pd

from Bio     import SeqIO
from Bio.Seq import Seq, MutableSeq

# ===== Defining functions =====

def determine_cds(gene_id, gff_file_path):
    df_gff = allel.gff3_to_dataframe(gff_file_path, attributes = ["ID", "Name", "Parent"])
    chrom  = df_gff.loc[df_gff["ID"] == gene_id,  "seqid"].values[0]
    strand = df_gff.loc[df_gff["ID"] == gene_id, "strand"].values[0]
    
    df_cds = df_gff.loc[
        (df_gff.ID.str.startswith(f"{gene_id}.1")) &
        (df_gff.type == "CDS") &
        (df_gff.Parent.str.endswith(".1"))
    ]
    
    return {
        "chrom"  : chrom,
        "strand" : strand,
        "starts" : df_cds.start.values,
        "ends"   : df_cds.end.values,
    }

def call_haplotype(samples, cds_data, vcf_file_format, ref_genome_file_path):
    
    chrom  = cds_data["chrom"]
    strand = cds_data["strand"]
    starts = cds_data["starts"]
    ends   = cds_data["ends"]
    
    assert len(starts) == len(ends), "starts and ends must be the same length"
        
    exon_offsets = [0] + list(np.cumsum( np.array(ends) - np.array(starts) + 1) )[:-1]
    
    ref_chrom_seq = [record.seq for record in SeqIO.parse(ref_genome_file_path, "fasta") if record.id == chrom][0]
    ref_sequences = [str(ref_chrom_seq[ start-1 : end ]) for start, end in zip(starts, ends)]
    
    ref_nucleotide_haplotype = "".join(ref_sequences)
    ref_sequence = Seq(ref_nucleotide_haplotype)

    ref_aa_haplotype = (str(ref_sequence.translate(to_stop = True)) if strand == "+" else str(ref_sequence.reverse_complement(inplace = False).translate(to_stop = True)))

    # ===== Parsing VCF =====
    vcf_reader = vcfpy.Reader.from_path(vcf_file_format % chrom)
    
    sample_sequences     = collections.OrderedDict()
    sample_offsets       = collections.OrderedDict()
    PID                  = collections.OrderedDict()
    non_phased_het_seen  = collections.OrderedDict()
    sample_unphaseable   = collections.OrderedDict()
    aa_haplotype         = collections.OrderedDict()
    nucleotide_haplotype = collections.OrderedDict()
    ns_changes           = collections.OrderedDict()
    num_missing          = collections.OrderedDict()
    num_called           = collections.OrderedDict()
    swap_phasing         = collections.OrderedDict()
    first_PGT            = collections.OrderedDict()
    
    for sample in samples:
        sample_sequences[sample]     = [MutableSeq(ref_sequence), MutableSeq(ref_sequence)]
        sample_offsets[sample]       = [0, 0]
        PID[sample]                  = ""
        non_phased_het_seen[sample]  = False
        sample_unphaseable[sample]   = ""
        aa_haplotype[sample]         = ""
        nucleotide_haplotype[sample] = ""
        num_missing[sample]          = 0
        num_called[sample]           = 0
        swap_phasing[sample]         = False
        first_PGT[sample]            = ""

    print("Starting VCF processing")
    # ===== Using VCF to make nucleotide haplotypes =====
    for start, end, exon_offset in zip(starts, ends, exon_offsets):
        for record in vcf_reader.fetch(chrom, start-1, end):
            if record.POS < start:
                continue
            
            if "PASS" not in record.FILTER:
                continue
            
            for sample in samples:
                call = record.call_for_sample[sample]
            
                GT  = call.data.get("GT")
                AD  = call.data.get("AD")
                POS = record.POS - start + exon_offset
                if GT == "./." or GT == ".|.":
                    num_missing[sample] += 1
                    continue # Skip processing missing calls
            
                num_called[sample] += 1
                
                if GT == "0/0" or GT == "0|0":
                    continue # Skip processing ref genotypes
                
                first_allele_GT = GT[0]
                second_allele_GT = GT[2]
                
                alleles = [record.REF] + [alt.value for alt in record.ALT]
                REF_len = len(record.REF) # Find lengths of alleles (to determine which are indels)
                
                first_allele_ALT_len  = len(alleles[int(first_allele_GT)])
                second_allele_ALT_len = len(alleles[int(second_allele_GT)])

                if (REF_len != first_allele_ALT_len or REF_len != second_allele_ALT_len):
                    continue # Skip processing indels
                
                # The following `if` section handles heterozygous calls at non-spanning deletion (* means spanning deletion)
                if (first_allele_GT != second_allele_GT) and (alleles[int(first_allele_GT)] != "*") and (alleles[int(second_allele_GT)] != "*"):
                    if "PGT" in call.data:
                        is_unphased = call.data.get("PGT") is None
                    else:
                        is_unphased = True
                    
                    if non_phased_het_seen[sample]:
                        sample_unphaseable[sample] = "*"
                    
                    if is_unphased:
                        non_phased_het_seen[sample] = True
                    else:
                        if PID[sample] == "":
                            PID[sample] = call.data.get("PID")
                            first_PGT[sample] = call.data.get("PGT")
                            
                            if AD[int(second_allele_GT)] > AD[int(first_allele_GT)]:
                                swap_phasing[sample] = True
                            
                        else:
                            if call.data.get("PID") != PID[sample]:
                                sample_unphaseable[sample] = "*"
                                
                    # Here we use phased genotype (PGT) if in same phasing group as first het
                    # if "PGT" in call.data and ( PID[sample] == call.data.get("PID") ): # In phasing group with first het call
                    if "PID" in call.data and ( PID[sample] == call.data.get("PID") ): # In phasing group with first het call
                        if first_PGT[sample] == call.data.get("PGT"):
                            if swap_phasing[sample]:
                                GT = second_allele_GT + "/" + first_allele_GT
                        else:
                            if not swap_phasing[sample]:
                                GT = second_allele_GT + "/" + first_allele_GT
                        
                    # Else we ensure we are taking the majority haplotype as the first allele
                    elif AD[int(second_allele_GT)] > AD[int(first_allele_GT)]:
                        GT = second_allele_GT + "/" + first_allele_GT

                for i, sample_offset in enumerate(sample_offsets[sample]): # The `i` only ever takes up values 0 or 1
                    alleles   = [record.REF] + [alt.value for alt in record.ALT]
                    GTint     = int(GT[i*2]) # Using the `i` value to either grab the first or second allele in, e.g., "0/2"
                    REF_len   = len(record.REF)
                    ALT_len   = len(alleles[GTint])

                    # Bugfix for accidental frameshifts: The following is required when the alleles are longer than the length of the remaining sequence, e.g. at the end of an exon
                    # Note this can make the REF_len and ALT_len values negative,
                    if REF_len > ( exon_offset + end - (start+POS) + 1 ):
                        REF_len = exon_offset + end - (start+POS) + 1
                        
                        if ALT_len > REF_len:
                            ALT_len = REF_len
                    
                    if (
                        GTint != 0 and            # If it's a non-ref allele
                        alleles[GTint] != "*" and # and it's not a spanning deletion
                        REF_len == ALT_len        # and it's a SNP, then:
                    ):  
                        sample_sequences[sample][i][
                            POS + sample_offsets[sample][i] : POS + sample_offsets[sample][i] + REF_len
                        ] = MutableSeq(str(alleles[GTint])[:ALT_len])
                        
                        sample_offsets[sample][i] = sample_offset + ( len(alleles[GTint]) - len(record.REF) )

    print("Building haplotypes")
    # ===== Forming amino acid haplotypes =====
    for sample in samples:
        if strand == "+":
            aa_0 = ( str(sample_sequences[sample][0].translate(to_stop = True)) if len(sample_sequences[sample][0]) % 3 == 0 else "!" )
            aa_1 = ( str(sample_sequences[sample][1].translate(to_stop = True)) if len(sample_sequences[sample][1]) % 3 == 0 else "!" )
        else:
            aa_0 = ( str(sample_sequences[sample][0].reverse_complement(inplace = False).translate(to_stop = True)) if len(sample_sequences[sample][0]) % 3 == 0 else "!" )
            aa_1 = ( str(sample_sequences[sample][1].reverse_complement(inplace = False).translate(to_stop = True)) if len(sample_sequences[sample][1]) % 3 == 0 else "!" )
        
        # In Pf7, we have some in-frame indels, hence we also need to check for these, and give "!" if indels exist
        if aa_0 != "!" and aa_1 != "!" and (len(aa_0) == len(aa_1) == len(ref_aa_haplotype)):
            if aa_0 == aa_1:
                ns_changes_list = []
                for i in range(len(ref_aa_haplotype)):
                    if aa_0[i] != ref_aa_haplotype[i]:
                        ns_changes_list.append(f"{ref_aa_haplotype[i]}{i+1}{aa_0[i]}")
                ns_changes[sample] = "/".join(ns_changes_list)
                
            else:
                ns_changes_list_0 = []
                for i in range(len(ref_aa_haplotype)):
                    if aa_0[i] != ref_aa_haplotype[i]:
                        ns_changes_list_0.append(f"{ref_aa_haplotype[i]}{i+1}{aa_0[i]}")
                
                ns_changes_0 = "/".join(ns_changes_list_0)
                
                ns_changes_list_1 = []
                for i in range(len(ref_aa_haplotype)):
                    if aa_1[i] != ref_aa_haplotype[i]:
                        ns_changes_list_1.append(f"{ref_aa_haplotype[i]}{i+1}{aa_1[i]}")

                ns_changes_1 = "/".join(ns_changes_list_1)

                if ns_changes_0 == "":
                    ns_changes[sample] = ns_changes_1.lower()
                elif ns_changes_1 == "":
                    ns_changes[sample] = ns_changes_0.lower()
                else:
                    ns_changes[sample] = (",".join([ns_changes_0, ns_changes_1])).lower()
        
        else:
            ns_changes[sample] = "!"

        # ===== populate nucleotide_haplotype and aa_haplotype =====
        if str(sample_sequences[sample][0]).upper() == str(sample_sequences[sample][1]).upper():
            nucleotide_haplotype[sample] = str(sample_sequences[sample][0])
            
            if strand == "+":
                aa_haplotype[sample] = ( str(sample_sequences[sample][0].translate(to_stop = True)) if len(sample_sequences[sample][0]) % 3 == 0 else "!" )
            else:
                aa_haplotype[sample] = ( str(sample_sequences[sample][0].reverse_complement(inplace = False).translate(to_stop = True)) if len(sample_sequences[sample][0]) % 3 == 0 else "!" )

        else:
            nucleotide_haplotype[sample] = f"{str(sample_sequences[sample][0])},{str(sample_sequences[sample][1])}"
            
            if strand == "+":
                aa_haplotype[sample] = f'{( str(sample_sequences[sample][0].translate(to_stop = True)) if len(sample_sequences[sample][0]) % 3 == 0 else "!" )},{( str(sample_sequences[sample][1].translate(to_stop = True)) if len(sample_sequences[sample][1]) % 3 == 0 else "!" )}'
            else:
                aa_haplotype[sample] = f'{( str(sample_sequences[sample][0].reverse_complement(inplace = False).translate(to_stop = True)) if len(sample_sequences[sample][0]) % 3 == 0 else "!" )},{( str(sample_sequences[sample][1].reverse_complement(inplace = False).translate(to_stop = True)) if len(sample_sequences[sample][1]) % 3 == 0 else "!" )}'

        # Add asterisk to end if sample is unphaseable
        ns_changes[sample]           += sample_unphaseable[sample]
        aa_haplotype[sample]         += sample_unphaseable[sample]
        nucleotide_haplotype[sample] += sample_unphaseable[sample]

        # Set ns_changes output to missing ('-') if no NS changes and at least one missing genotype call in the region
        if (ns_changes[sample] == "" or ns_changes[sample] == "*") and num_missing[sample] > 0:
            ns_changes[sample] = "-"
                
        # Set sequence output to missing ('-') if any missing genotype call in the gene
        if num_missing[sample] > 0:
            aa_haplotype[sample]         = "-"
            nucleotide_haplotype[sample] = "-"
    
    df_all_haplotypes = pd.DataFrame({
        "aa_haplotype"         : pd.Series(aa_haplotype),
        "nucleotide_haplotype" : pd.Series(nucleotide_haplotype),
        "ns_changes"           : pd.Series(ns_changes),
    })

    print("Finished")

    return {
        "df_all_haplotypes"        : df_all_haplotypes,
        "ref_aa_haplotype"         : ref_aa_haplotype,
        "ref_nucleotide_haplotype" : ref_nucleotide_haplotype,
    }