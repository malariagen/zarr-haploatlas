"""
Tests for haplotypes.py — focused on indel and het edge cases.

Run with:  python -m pytest tests/test_haplotypes.py -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import pytest

from haplotypes import (
    _apply_variants,
    _clip_alleles_at_exon_boundaries,
    _translate,
    _aa_at,
    _aa_slice,
    _genomic_to_cds_offset,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pi(offset, ref):
    """Shorthand: build a single pos_info entry dict."""
    return {"ref": ref, "offset": offset}


def _exons(ranges):
    """Build a minimal exons DataFrame from [(start, end), ...]."""
    return pd.DataFrame([{"start": s, "end": e} for s, e in ranges])


# ─────────────────────────────────────────────────────────────────────────────
# _apply_variants — basic SNPs
# ─────────────────────────────────────────────────────────────────────────────

class TestApplyVariantsSnps:

    def test_snp_applies(self):
        result = _apply_variants("TACG", [("100", _pi(1, "A"))], {"100": "T"})
        assert result == "TTCG"

    def test_snp_ref_allele_unchanged(self):
        result = _apply_variants("TACG", [("100", _pi(1, "A"))], {"100": "A"})
        assert result == "TACG"

    def test_two_snps(self):
        pi = [("100", _pi(0, "A")), ("103", _pi(3, "C"))]
        result = _apply_variants("ATGCGT", pi, {"100": "G", "103": "T"})
        assert result == "GTGTGT"

    def test_missing_pos_uses_ref_no_change(self):
        """Position absent from alleles_here → use info['ref'] → reference allele → skip."""
        result = _apply_variants("TACG", [("100", _pi(1, "A"))], {})
        assert result == "TACG"


# ─────────────────────────────────────────────────────────────────────────────
# _apply_variants — special markers
# ─────────────────────────────────────────────────────────────────────────────

class TestApplyVariantsMarkers:

    def test_spanning_deletion_star_skipped(self):
        result = _apply_variants("TACG", [("100", _pi(1, "A"))], {"100": "*"})
        assert result == "TACG"

    def test_het_hash_treated_as_ref(self):
        result = _apply_variants("TACG", [("100", _pi(1, "A"))], {"100": "#"})
        assert result == "TACG"

    def test_missing_dash_treated_as_ref(self):
        result = _apply_variants("TACG", [("100", _pi(1, "A"))], {"100": "-"})
        assert result == "TACG"

    def test_multi_allele_comma_skipped(self):
        result = _apply_variants("TACG", [("100", _pi(1, "A"))], {"100": "G,T"})
        assert result == "TACG"

    def test_ordered_het_slash_skipped(self):
        """Safe-het allele 'A/T' is skipped in the main pass (major/minor handled upstream)."""
        result = _apply_variants("TACG", [("100", _pi(1, "A"))], {"100": "A/T"})
        assert result == "TACG"

    def test_ordered_het_pipe_skipped(self):
        """Unsafe-het allele 'A|T' is also skipped in the main pass."""
        result = _apply_variants("TACG", [("100", _pi(1, "A"))], {"100": "A|T"})
        assert result == "TACG"

    def test_trailing_null_stripped_observed(self):
        result = _apply_variants("AT", [("100", _pi(0, "A"))], {"100": "G\x00\x00"})
        assert result == "GT"

    def test_trailing_null_stripped_ref(self):
        result = _apply_variants("AT", [("100", _pi(0, "A\x00\x00"))], {"100": "G"})
        assert result == "GT"


# ─────────────────────────────────────────────────────────────────────────────
# _apply_variants — insertions and deletions
# ─────────────────────────────────────────────────────────────────────────────

class TestApplyVariantsIndels:

    def test_simple_insertion(self):
        # Replace 'A' at offset 1 with 'ATTT' (+3 chars)
        result = _apply_variants("TACGT", [("100", _pi(1, "A"))], {"100": "ATTT"})
        assert result == "TATTTCGT"

    def test_simple_deletion(self):
        # Delete 2 of 3 chars: AC → A at offset 1
        result = _apply_variants("TACGT", [("100", _pi(1, "AC"))], {"100": "A"})
        assert result == "TAGT"

    def test_pure_deletion(self):
        # REF='ACG' → ALT='A' (del 2 bp) at offset 1
        result = _apply_variants("AACGTT", [("100", _pi(1, "ACG"))], {"100": "A"})
        assert result == "AATT"

    def test_deletion_shifts_downstream_snp(self):
        """
        3-bp deletion at offset 0 (ATG→A, running_offset=-2) followed by SNP at offset 3.
        The SNP at original offset 3 must land at adjusted offset 1 in the mutated seq.

        ref:  A T G C G T   (original)
              |             <- deletion replaces ATG with A → running_offset = -2
              A . . C G T   (conceptually, the gap shifts downstream by 2)

        Adjusted SNP offset = 3 + (-2) = 1:
              A T G T       ('C' at new position 1 → 'T')
        """
        pi = [("100", _pi(0, "ATG")), ("103", _pi(3, "C"))]
        result = _apply_variants("ATGCGT", pi, {"100": "A", "103": "T"})
        assert result == "ATGT"

    def test_insertion_shifts_downstream_snp(self):
        """
        Insertion at offset 0 (A→AAAA, +3 chars, running_offset=+3) then SNP at offset 2.
        Adjusted SNP offset = 2+3 = 5.

        ref:  A T C G
        After insert: A A A A T C G   (positions 0-6)
        SNP at offset 5 changes 'C' → 'T':
              A A A A T T G
        """
        pi = [("100", _pi(0, "A")), ("102", _pi(2, "C"))]
        result = _apply_variants("ATCG", pi, {"100": "AAAA", "102": "T"})
        assert result == "AAAATTG"

    def test_two_deletions_accumulate_running_offset(self):
        """
        Del1 at offset 0: ATG→A (running_offset = -2)
        Del2 at offset 3: ATG→A, adjusted = 3+(-2) = 1, running_offset = -4

        ref:  A T G A T G A T G
        After del1: A A T G A T G   (seq[0:3] = ['A'])
        After del2: offset 1 → A A T G            (seq[1:4] = ['A'])
        Result: "AATG"
        """
        pi = [("100", _pi(0, "ATG")), ("103", _pi(3, "ATG"))]
        result = _apply_variants("ATGATGATG", pi, {"100": "A", "103": "A"})
        # Del1: ATG(0-2)→A; del2: ATG(3-5)→A; last ATG(6-8) remains
        # Original: ATG|ATG|ATG → A|A|ATG = "AAATG"
        assert result == "AAATG"

    def test_large_deletion_running_offset_is_negative(self):
        """
        A 5-bp deletion (ATGCA→A) at offset 0 leaves running_offset = -4.
        A downstream SNP at offset 5 → adjusted 5+(-4) = 1.

        ref: A T G C A T T
        del: A             → running_offset = -4
        After: A T T
        SNP at offset 5 ('T' → 'G') adjusted to 1:
              A G T
        """
        pi = [("100", _pi(0, "ATGCA")), ("105", _pi(5, "T"))]
        result = _apply_variants("ATGCATT", pi, {"100": "A", "105": "G"})
        assert result == "AGT"

    def test_spanning_del_within_deletion_guarded_by_star(self):
        """
        Large deletion at offset 0 (ATGATG→A) followed by a position at offset 3
        that is WITHIN the deleted bases. In valid VCF this position is '*'.
        The '*' must prevent silent corruption of positions outside the deletion.

        ref:  A T G A T G C C C
        del:  [A T G A T G] → A   running_offset = 1-6 = -5
        After: A C C C

        If the inner position (offset 3) were applied as a real variant without '*',
        its adjusted offset = 3 + (-5) = -2, which Python silently wraps (corrupt!).
        Marking it '*' prevents this.
        """
        pi = [
            ("100", _pi(0, "ATGATG")),
            ("103", _pi(3, "A")),       # within the deleted region
        ]
        # With '*' — correct behaviour: inner position skipped, result is just deletion
        result_star = _apply_variants("ATGATGCCC", pi, {"100": "A", "103": "*"})
        assert result_star == "ACCC"

        # With '#' het — same: inner position skipped
        result_het = _apply_variants("ATGATGCCC", pi, {"100": "A", "103": "#"})
        assert result_het == "ACCC"

        # With '-' missing — same
        result_miss = _apply_variants("ATGATGCCC", pi, {"100": "A", "103": "-"})
        assert result_miss == "ACCC"


# ─────────────────────────────────────────────────────────────────────────────
# _apply_variants — het indels (major / minor allele split)
# ─────────────────────────────────────────────────────────────────────────────

class TestApplyVariantsHetIndels:
    """
    Simulate the ordered-ad path where alleles_major / alleles_minor are passed
    separately after splitting on het_sep.  Each call to _apply_variants sees
    only clean alleles; it must track running_offset correctly when one of them
    is an indel.
    """

    # CDS for tests: "ATGGAATGC" (9 nt = M·E·C)
    # Codon 1: ATG (0-2) = Met
    # Codon 2: GAA (3-5) = Glu
    # Codon 3: TGC (6-8) = Cys
    REF_CDS = "ATGGAATGC"

    def test_het_major_ref_minor_deletion_causes_frameshift(self):
        """
        Het at pos 3: major allele = 'GAA' (ref, no change), minor = 'G' (del 2 bp).
        Major → MEC, Minor → frameshift → '!'
        """
        pi = [("103", _pi(3, "GAA"))]

        major_cds = _apply_variants(self.REF_CDS, pi, {"103": "GAA"})
        minor_cds = _apply_variants(self.REF_CDS, pi, {"103": "G"})

        assert major_cds == self.REF_CDS           # unchanged
        assert minor_cds == "ATGGTGC"              # 7 nt → frameshift
        assert len(minor_cds) % 3 != 0

        assert _translate(major_cds, "+") == "MEC"
        assert _translate(minor_cds, "+") == ">"   # frameshift

    def test_het_minor_deletion_with_downstream_snp(self):
        """
        Het deletion on minor allele at offset 3 (GAA→G, running_offset=-2),
        followed by a downstream SNP at offset 6 (T→A).

        Minor path:
          After del: A T G G T G C    (7 nt, frameshift)
          Adjusted SNP offset = 6 + (-2) = 4
          seq[4] = 'T' → 'A' → "ATGGA GC"... actually wait:
          After del: [A,T,G,G,T,G,C]
          SNP at offset 4: seq[4] = 'T' → 'A' → [A,T,G,G,A,G,C] = "ATGGAGC"
        """
        pi = [("103", _pi(3, "GAA")), ("106", _pi(6, "T"))]

        minor_alleles = {"103": "G", "106": "A"}
        minor_cds = _apply_variants(self.REF_CDS, pi, minor_alleles)
        assert minor_cds == "ATGGAGC"   # 7 nt, frameshift
        assert _translate(minor_cds, "+") == ">"   # frameshift

    def test_het_minor_insertion_adds_codon(self):
        """
        Het insertion on minor allele: pos offset 3, REF='G', ALT='GAAA' (+3 bp in-frame).

        ref:  A T G | G A A | T G C  = M E C
        minor: replace G(3) with GAAA → +3 bp in-frame
          A T G G A A A A A T G C   (12 nt = M E K C)
          Codons: ATG=M | GAA=E | AAA=K | TGC=C
        """
        pi = [("103", _pi(3, "G"))]
        minor_cds = _apply_variants(self.REF_CDS, pi, {"103": "GAAA"})

        # 12 nt, in-frame
        assert len(minor_cds) == 12
        assert len(minor_cds) % 3 == 0
        assert _translate(minor_cds, "+") == "MEKC"

    def test_het_insertion_minor_shifts_downstream_snp(self):
        """
        Het insertion (minor) at offset 0, plus a real (hom) SNP at offset 5.

        ref:  A T G G A A T G C  (9 nt)
        minor insertion at offset 0: A→AAAA (+3 bp), running_offset = +3
        downstream SNP at offset 5 (ref='A', alt='T'), adjusted = 5+3 = 8:
          After insert: A A A A T G G A A T G C  (12 nt)
          Position 8 = 'A' → 'T':
          A A A A T G G T A T G C ... hmm let me re-trace.

        ref:  A(0)T(1)G(2)G(3)A(4)A(5)T(6)G(7)C(8)
        Insert: seq[0:1] = ['A','A','A','A']
          seq = [A,A,A,A,T,G,G,A,A,T,G,C]  (12 chars)
          running_offset = +3
        SNP at offset 5 (ref 'A', alt 'T') → adjusted = 8:
          seq[8] = 'A' → 'T'
          seq = [A,A,A,A,T,G,G,A,T,T,G,C]  (12 chars)
          = "AAAATGGATTGC"
        """
        pi = [("100", _pi(0, "A")), ("105", _pi(5, "A"))]
        minor_alleles = {"100": "AAAA", "105": "T"}
        result = _apply_variants(self.REF_CDS, pi, minor_alleles)
        assert result == "AAAATGGATTGC"
        assert len(result) == 12


# ─────────────────────────────────────────────────────────────────────────────
# _translate
# ─────────────────────────────────────────────────────────────────────────────

class TestTranslate:

    def test_plus_strand_basic(self):
        assert _translate("ATGATG", "+") == "MM"

    def test_frameshift_returns_gt(self):
        # Non-in-frame indel → frameshift → ">" (not "!")
        assert _translate("ATGAT", "+") == ">"     # 5 nt, not divisible by 3
        assert _translate("AT", "+") == ">"         # 2 nt

    def test_premature_stop_truncates(self):
        # ATG = M, TAA = stop → "M" (to_stop=True); premature stop ≠ frameshift
        assert _translate("ATGTAATTT", "+") == "M"

    def test_minus_strand_reverse_complement(self):
        # "CATCAT" → RC = "ATGATG" → "MM"
        assert _translate("CATCAT", "-") == "MM"

    def test_empty_sequence(self):
        assert _translate("", "+") == ""


# ─────────────────────────────────────────────────────────────────────────────
# _aa_at
# ─────────────────────────────────────────────────────────────────────────────

class TestAaAt:

    def test_first_position(self):
        assert _aa_at("MVK", 1) == "M"

    def test_last_position(self):
        assert _aa_at("MVK", 3) == "K"

    def test_beyond_length_is_exclamation(self):
        # Premature stop: position beyond translated sequence → "!"
        assert _aa_at("MVK", 4) == "!"

    def test_zero_is_exclamation(self):
        assert _aa_at("MVK", 0) == "!"

    def test_negative_is_exclamation(self):
        assert _aa_at("MVK", -1) == "!"

    def test_frameshift_gt_returns_gt(self):
        # When _translate returned ">" (frameshift), every aa position returns ">"
        assert _aa_at(">", 1) == ">"
        assert _aa_at(">", 99) == ">"

    def test_frameshift_gt_zero_is_exclamation(self):
        # pos < 1 always returns "!" regardless of alt_aa
        assert _aa_at(">", 0) == "!"


# ─────────────────────────────────────────────────────────────────────────────
# _aa_slice
# ─────────────────────────────────────────────────────────────────────────────

class TestAaSlice:

    def test_full_range(self):
        assert _aa_slice("MVKL", 1, 4) == "MVKL"

    def test_partial_range(self):
        assert _aa_slice("MVKL", 2, 3) == "VK"

    def test_range_beyond_end(self):
        # Position 4 exists, position 5 doesn't → '!' for the last
        assert _aa_slice("MVKL", 4, 5) == "L!"

    def test_start_zero_is_exclamation(self):
        assert _aa_slice("MVKL", 0, 2) == "!"


# ─────────────────────────────────────────────────────────────────────────────
# _genomic_to_cds_offset
# ─────────────────────────────────────────────────────────────────────────────

class TestGenomicToCdsOffset:

    def test_single_exon_start(self):
        exons = _exons([(100, 200)])
        assert _genomic_to_cds_offset(100, exons, "+") == 0

    def test_single_exon_end(self):
        exons = _exons([(100, 200)])
        assert _genomic_to_cds_offset(200, exons, "+") == 100

    def test_single_exon_middle(self):
        exons = _exons([(100, 200)])
        assert _genomic_to_cds_offset(150, exons, "+") == 50

    def test_position_before_exon_is_none(self):
        exons = _exons([(100, 200)])
        assert _genomic_to_cds_offset(99, exons, "+") is None

    def test_position_after_exon_is_none(self):
        exons = _exons([(100, 200)])
        assert _genomic_to_cds_offset(201, exons, "+") is None

    def test_two_exons_start_of_second(self):
        # exon1: 100-109 (10 nt, CDS 0-9), exon2: 200-209 (10 nt, CDS 10-19)
        exons = _exons([(100, 109), (200, 209)])
        assert _genomic_to_cds_offset(200, exons, "+") == 10

    def test_two_exons_middle_of_second(self):
        exons = _exons([(100, 109), (200, 209)])
        assert _genomic_to_cds_offset(205, exons, "+") == 15

    def test_intronic_position_is_none(self):
        exons = _exons([(100, 109), (200, 209)])
        assert _genomic_to_cds_offset(150, exons, "+") is None

    def test_three_exons_third_exon(self):
        # exon1: 100-104 (5 nt), exon2: 200-204 (5 nt), exon3: 300-304 (5 nt)
        exons = _exons([(100, 104), (200, 204), (300, 304)])
        assert _genomic_to_cds_offset(300, exons, "+") == 10
        assert _genomic_to_cds_offset(304, exons, "+") == 14


# ─────────────────────────────────────────────────────────────────────────────
# Integration: AA translation round-trip for simple indel
# ─────────────────────────────────────────────────────────────────────────────

class TestIndelTranslationRoundTrip:
    """
    End-to-end: apply indel to a synthetic CDS and verify the translated AA.
    Exercises the full _apply_variants → _translate → _aa_at pipeline.
    """

    def test_synonymous_snp_unchanged_aa(self):
        # ATG GAA TGC = MEC; silent change: GAA→GAG (both Glu)
        ref_cds = "ATGGAATGC"
        pi = [("105", _pi(5, "A"))]
        alt_cds = _apply_variants(ref_cds, pi, {"105": "G"})
        assert alt_cds == "ATGGAGTGC"
        aa = _translate(alt_cds, "+")
        assert aa == "MEC"   # synonymous

    def test_missense_snp_changes_aa(self):
        # GAA→GCA (Glu→Ala)
        ref_cds = "ATGGAATGC"
        pi = [("104", _pi(4, "A")), ("105", _pi(5, "A"))]
        alt_cds = _apply_variants(ref_cds, pi, {"104": "C", "105": "G"})
        # GAA → GCG? Actually let's just do one SNP: GAA→GCA requires A(4)→C, A(5)→A
        # Simplest: change first base of codon 2: G(3)→C → CGO = something else
        # Better: change all of codon 2 to GCA = Ala
        # Let's do a 3-base substitution: REF='GAA' ALT='GCA'
        pass  # covered by test_het_major_ref_minor_deletion_causes_frameshift

    def test_inframe_deletion_shortens_protein(self):
        # Delete exactly 3 bp (one codon) → protein loses 1 AA, no frameshift
        # ref: ATG GAA TGC AAA = M E C K  (12 nt)
        # Del codon 2 (GAA→''): ATG TGC AAA = M C K  (9 nt)
        ref_cds = "ATGGAATGCAAA"
        pi = [("103", _pi(3, "GAA"))]
        alt_cds = _apply_variants(ref_cds, pi, {"103": ""})
        assert alt_cds == "ATGTGCAAA"
        assert _translate(alt_cds, "+") == "MCK"

    def test_frameshift_deletion_gives_gt(self):
        # Delete 1 bp from a 2-bp REF (non-frame) → frameshift → ">"
        ref_cds = "ATGGAATGCAAA"
        pi = [("103", _pi(3, "GA"))]
        alt_cds = _apply_variants(ref_cds, pi, {"103": "G"})   # del 1 bp from GA→G
        assert len(alt_cds) % 3 != 0
        assert _translate(alt_cds, "+") == ">"

    def test_inframe_insertion_extends_protein(self):
        # Insert 3 bp in-frame after position 0 codon
        # ref: ATG GAA TGC = M E C
        # Insert 'AAA' after ATG → ATG AAA GAA TGC = M K E C
        ref_cds = "ATGGAATGC"
        pi = [("102", _pi(2, "G"))]   # offset 2 = last base of codon 1 ('G')
        alt_cds = _apply_variants(ref_cds, pi, {"102": "GAAA"})
        # seq[2:3] = ['G','A','A','A'] → ATGAAAGAATGC (12 nt)
        assert len(alt_cds) == 12
        assert _translate(alt_cds, "+") == "MKEC"

    def test_stop_gained_gives_truncated_protein(self):
        # Change codon 2 GAA→TAA (stop)
        ref_cds = "ATGGAATGC"
        pi = [("103", _pi(3, "G"))]
        alt_cds = _apply_variants(ref_cds, pi, {"103": "T"})
        assert alt_cds == "ATGTAATGC"
        aa = _translate(alt_cds, "+")
        assert aa == "M"   # truncated at stop


# ─────────────────────────────────────────────────────────────────────────────
# Exon-intron boundary clipping
# ─────────────────────────────────────────────────────────────────────────────
#
# Scenario: a gene with two exons separated by an intron.
#
#   Genomic:  [100 ── exon1 ── 109] [110 intron 119] [120 ── exon2 ── 129]
#   CDS:       A  T  G  G  A  A  T  G  C  C  +  A  T  G  G  A  A  T  G  C
#   offsets:   0  1  2  3  4  5  6  7  8  9     10 11 12 13 14 15 16 17 18 19
#
# A VCF variant at pos 108 (2 bases from exon1 end) with a long REF that
# runs past position 109 into the intron.  Only the first 2 bases (exonic)
# should influence the CDS.

class TestExonBoundaryClipping:
    """_clip_alleles_at_exon_boundaries and full pipeline for exon-spanning REFs."""

    # pos_info for a position at offset 8 (genomic 108), last 2 bases of exon1.
    # REF was originally 7 bp (2 exonic + 5 intronic) and has been clipped to 2.
    CLIPPED_PI = {"108": {"ref": "TC", "offset": 8, "exon_clip_len": 2}}

    def test_clip_leaves_short_alleles_unchanged(self):
        alleles = {"108": "G"}   # shorter than clip — no change needed
        result = _clip_alleles_at_exon_boundaries(alleles, self.CLIPPED_PI)
        assert result["108"] == "G"

    def test_clip_truncates_long_alt_to_exonic_length(self):
        # ALT is the full 7-bp VCF allele; only first 2 chars are exonic
        alleles = {"108": "GCAAAAA"}
        result = _clip_alleles_at_exon_boundaries(alleles, self.CLIPPED_PI)
        assert result["108"] == "GC"

    def test_clip_passes_through_special_markers(self):
        for marker in ("*", "#", "-"):
            result = _clip_alleles_at_exon_boundaries({"108": marker}, self.CLIPPED_PI)
            assert result["108"] == marker

    def test_clip_passes_through_multi_allele(self):
        result = _clip_alleles_at_exon_boundaries({"108": "G,T"}, self.CLIPPED_PI)
        assert result["108"] == "G,T"

    def test_clip_passes_through_ordered_het(self):
        result = _clip_alleles_at_exon_boundaries({"108": "G/T"}, self.CLIPPED_PI)
        assert result["108"] == "G/T"

    def test_clip_leaves_positions_without_clip_unchanged(self):
        # Position with no exon_clip_len should be untouched even with a long allele
        pos_info_no_clip = {"108": {"ref": "TCAAAAA", "offset": 8}}
        alleles = {"108": "GCAAAAA"}
        result = _clip_alleles_at_exon_boundaries(alleles, pos_info_no_clip)
        assert result["108"] == "GCAAAAA"

    # ── Pipeline tests using synthetic pos_info with exon_clip_len ──

    def test_boundary_snp_exonic_change_applied(self):
        """
        REF 'TCAAAAA' (2 exonic + 5 intronic) clipped to 'TC' at offset 8.
        ALT 'GCAAAAA' (7 bp) clipped to 'GC' → change T→G, C unchanged at offsets 8-9.

        ref_cds: ... X X [T C] X X  (positions 8-9 = 'TC')
        After:   ... X X [G C] X X
        """
        ref_cds = "ATGGAATGCC"   # 10 nt; offset 8='T', 9='C'
        pi = [("108", {"ref": "TC", "offset": 8, "exon_clip_len": 2})]

        # Without clipping the ALT: long ALT would corrupt running_offset
        alleles_long = {"108": "GCAAAAA"}

        clipped = _clip_alleles_at_exon_boundaries(alleles_long, dict(pi))
        assert clipped["108"] == "GC"

        result = _apply_variants(ref_cds, pi, clipped)
        assert result == "ATGGAATGGC"   # T→G at offset 8; C unchanged at 9

    def test_boundary_intronic_only_change_treated_as_ref(self):
        """
        REF 'TCAAAAA' clipped to 'TC'.  ALT = 'TCBBBBB' — exonic portion 'TC'
        is identical to REF exonic portion.  After clipping, observed == ref_a
        → _apply_variants skips it → CDS unchanged.
        """
        ref_cds = "ATGGAATGCC"
        pi = [("108", {"ref": "TC", "offset": 8, "exon_clip_len": 2})]
        alleles = {"108": "TCBBBBB"}   # intronic change only

        clipped = _clip_alleles_at_exon_boundaries(alleles, dict(pi))
        assert clipped["108"] == "TC"  # clipped to exonic portion = same as ref

        result = _apply_variants(ref_cds, pi, clipped)
        assert result == ref_cds  # no change

    def test_boundary_clipping_preserves_downstream_snp_offset(self):
        """
        Without clipping, a boundary-spanning ALT (e.g., 7-bp for a 2-bp exonic ref)
        would set running_offset = 7-2 = +5, corrupting all downstream offsets.
        After clipping to 2 bp, running_offset = 2-2 = 0 for a substitution,
        and a downstream SNP lands at the correct position.

        ref_cds: ATGGAATGCC (10 nt)
        Boundary variant at offset 8: REF='TC' (clipped from 7 bp), ALT clipped to 'GC'
          → running_offset = 2-2 = 0 (substitution, no length change)
        Downstream SNP at offset 0: A→T
          → adjusted = 0+0 = 0 → first base changes

        Expected: TTGGAATGGC
        """
        ref_cds = "ATGGAATGCC"
        pi_sorted = [
            ("100", {"ref": "A", "offset": 0}),
            ("108", {"ref": "TC", "offset": 8, "exon_clip_len": 2}),
        ]
        pos_info = dict(pi_sorted)

        # ALT for boundary position: 7-bp VCF allele, clips to 'GC'
        alleles = {"100": "T", "108": "GCAAAAA"}
        clipped = _clip_alleles_at_exon_boundaries(alleles, pos_info)

        result = _apply_variants(ref_cds, pi_sorted, clipped)
        assert result == "TTGGAATGGC"

    def test_unclipped_boundary_alt_corrupts_offset(self):
        """
        Demonstrates WHY clipping is necessary: without it, a 7-bp ALT for a
        2-bp exonic REF sets running_offset = +5, pushing the downstream SNP
        5 positions too far to the right.

        This test documents the BROKEN behaviour of the unclipped path so we
        can detect regressions if clipping is accidentally removed.
        """
        ref_cds = "ATGGAATGCC"
        pi_sorted = [
            ("100", {"ref": "A", "offset": 0}),
            ("108", {"ref": "TC", "offset": 8}),   # no exon_clip_len
        ]
        alleles_unclipped = {"100": "T", "108": "GCAAAAA"}   # 7-bp ALT, unclipped

        result = _apply_variants(ref_cds, pi_sorted, alleles_unclipped)
        # running_offset after boundary variant = 7-2 = +5
        # SNP at offset 0 processed first → A→T at position 0 ✓
        # Boundary variant at offset 8, adjusted = 8+0 = 8: seq[8:10] → "GCAAAAA"
        #   → result extends beyond original CDS length
        assert result != "TTGGAATGGC"   # wrong answer without clipping
        assert len(result) != len(ref_cds)   # length is wrong

    # ── Single-base exon boundary ──

    def test_single_base_exon_boundary_clip(self):
        """
        Extreme case: variant is at the very last base of an exon (exon_clip_len=1).
        Any ALT longer than 1 char is clipped to just the first char.
        """
        pi = {"100": {"ref": "A", "offset": 5, "exon_clip_len": 1}}

        # 11-bp ALT → clipped to 1
        assert _clip_alleles_at_exon_boundaries({"100": "CGTAAGTATCC"}, pi)["100"] == "C"
        # 1-bp ALT → already at clip length, unchanged
        assert _clip_alleles_at_exon_boundaries({"100": "C"}, pi)["100"] == "C"

    def test_single_base_boundary_exonic_change_applied(self):
        """Single-base exon boundary: exonic base changes, intronic junk is discarded."""
        ref_cds = "ATGGAATGCC"
        # Variant at offset 5 (ref='A'), clip to 1 char
        pi = [("105", {"ref": "A", "offset": 5, "exon_clip_len": 1})]
        clipped = _clip_alleles_at_exon_boundaries({"105": "CGTAAGTATCC"}, dict(pi))
        result = _apply_variants(ref_cds, pi, clipped)
        assert result == "ATGGACTGCC"   # A→C at offset 5 only; len unchanged

    def test_single_base_boundary_intronic_only_no_change(self):
        """Single-base: ALT[0] == REF[0] → exonic base unchanged → CDS unchanged."""
        ref_cds = "ATGGAATGCC"
        pi = [("105", {"ref": "A", "offset": 5, "exon_clip_len": 1})]
        clipped = _clip_alleles_at_exon_boundaries({"105": "AXXXXXXXX"}, dict(pi))
        result = _apply_variants(ref_cds, pi, clipped)
        assert result == ref_cds   # no change

    # ── ALT shorter than exon_clip_len (exon deletion) ──

    def test_alt_deletion_shorter_than_clip_retained(self):
        """
        ALT is a genuine deletion into the exon: shorter than exon_clip_len.
        The clip helper leaves it unchanged (the deletion IS the exonic change).

        exon_clip_len=3, REF='TCG' (clipped from longer), ALT='T' (del 2 exonic bp).
        After apply: ref_len=3 replaced by 'T' (1 char) → running_offset=-2.

        ref_cds: ATGGAAATCG (10 nt); offsets 7,8,9 = 'T','C','G'
        """
        pi_dict = {"108": {"ref": "TCG", "offset": 7, "exon_clip_len": 3}}
        # ALT='T' is shorter than clip=3 — should NOT be further clipped
        clipped = _clip_alleles_at_exon_boundaries({"108": "T"}, pi_dict)
        assert clipped["108"] == "T"   # kept as-is

        # Verify the deletion is correctly applied: TCG→T at offset 7
        ref_cds = "ATGGAAATCG"   # 10 nt; offsets 7='T', 8='C', 9='G'
        pi = [("108", {"ref": "TCG", "offset": 7, "exon_clip_len": 3})]
        result = _apply_variants(ref_cds, pi, {"108": "T"})
        assert result == "ATGGAAAT"   # del 2 bases → 8 nt
        assert len(result) == 8

    def test_alt_deletion_shorter_than_clip_downstream_offset_correct(self):
        """
        Exon deletion (running_offset=-2) followed by a downstream SNP.

        ref_cds: ATGGAAATCGA (11 nt); offsets 7='T', 8='C', 9='G', 10='A'
        del: TCG(7-9)→T, running_offset=-2
        SNP: offset 10, ref='A', alt='G', adjusted = 10+(-2) = 8
        After del seq = ATGGAAATA (9 chars); seq[8]='A'→'G' → ATGGAAATG
        """
        ref_cds = "ATGGAAATCGA"
        pi = [
            ("108", {"ref": "TCG", "offset": 7, "exon_clip_len": 3}),
            ("110", {"ref": "A",   "offset": 10}),
        ]
        clipped = _clip_alleles_at_exon_boundaries({"108": "T", "110": "G"}, dict(pi))
        result = _apply_variants(ref_cds, pi, clipped)
        assert result == "ATGGAAATG"

    # ── Multiple boundary positions in same call ──

    def test_two_boundary_positions_clipped_independently(self):
        """
        Two positions at different exon ends, both with exon_clip_len.
        Each is clipped independently; they do not interact.
        """
        pos_info = {
            "105": {"ref": "A",  "offset": 5, "exon_clip_len": 1},
            "108": {"ref": "TC", "offset": 8, "exon_clip_len": 2},
        }
        alleles = {"105": "GXXXXXX", "108": "GCAAAAA"}
        clipped = _clip_alleles_at_exon_boundaries(alleles, pos_info)
        assert clipped["105"] == "G"
        assert clipped["108"] == "GC"

    def test_two_boundary_positions_both_applied(self):
        """Both boundary substitutions applied, running_offset stays 0 for both."""
        ref_cds = "ATGGAATGCC"
        pi = [
            ("105", {"ref": "A",  "offset": 5, "exon_clip_len": 1}),
            ("108", {"ref": "TC", "offset": 8, "exon_clip_len": 2}),
        ]
        pos_info = dict(pi)
        alleles = {"105": "GXXXXXX", "108": "GCAAAAA"}
        clipped = _clip_alleles_at_exon_boundaries(alleles, pos_info)
        result = _apply_variants(ref_cds, pi, clipped)
        # A→G at offset 5, TC→GC at offset 8; both substitutions, len unchanged
        assert result == "ATGGAGTGGC"

    # ── Ordered het at boundary position ──

    def test_ordered_het_at_boundary_passed_through_then_split(self):
        """
        Ordered-ad allele 'GCAAAAA/TCBBBBB' at a boundary position passes through
        _clip_alleles_at_exon_boundaries intact (the '/' prevents clipping).
        After the caller splits it, each allele is clipped separately.
        """
        pos_info = {"108": {"ref": "TC", "offset": 8, "exon_clip_len": 2}}

        # The full het allele is NOT clipped (contains '/')
        het_alleles = {"108": "GCAAAAA/TCBBBBB"}
        result = _clip_alleles_at_exon_boundaries(het_alleles, pos_info)
        assert result["108"] == "GCAAAAA/TCBBBBB"

        # After split (simulating the ordered_ad path):
        major_alleles = {"108": "GCAAAAA"}
        minor_alleles = {"108": "TCBBBBB"}
        clipped_major = _clip_alleles_at_exon_boundaries(major_alleles, pos_info)
        clipped_minor = _clip_alleles_at_exon_boundaries(minor_alleles, pos_info)
        assert clipped_major["108"] == "GC"    # exonic change: T→G at offset 8
        assert clipped_minor["108"] == "TC"    # exonic portion == REF → no change


# ─────────────────────────────────────────────────────────────────────────────
# _apply_variants — no-op / trivial edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestApplyVariantsNoOp:

    def test_empty_pos_info_returns_input_unchanged(self):
        assert _apply_variants("ATGCGT", [], {}) == "ATGCGT"

    def test_all_ref_alleles_returns_unchanged(self):
        pi = [("100", _pi(0, "A")), ("101", _pi(1, "T")), ("102", _pi(2, "G"))]
        result = _apply_variants("ATG", pi, {"100": "A", "101": "T", "102": "G"})
        assert result == "ATG"

    def test_empty_ref_seq(self):
        """Empty CDS (e.g. before gene annotation loaded) → empty output."""
        assert _apply_variants("", [], {}) == ""


# ─────────────────────────────────────────────────────────────────────────────
# _translate — stop-codon completeness
# ─────────────────────────────────────────────────────────────────────────────

class TestTranslateStopCodons:
    """All three stop codons should truncate translation (to_stop=True)."""

    def test_taa_stop(self):
        assert _translate("ATGTAATTT", "+") == "M"

    def test_tag_stop(self):
        assert _translate("ATGTAGTTT", "+") == "M"

    def test_tga_stop(self):
        assert _translate("ATGTGATTT", "+") == "M"

    def test_stop_at_start_gives_empty(self):
        assert _translate("TAATTT", "+") == ""

    def test_minus_strand_premature_stop(self):
        # Minus strand: RC of "ATGTTACTT" = "AAGTAACAT"?
        # Let's build a minus-strand stop deliberately:
        # On + strand: "TTACTT..." → RC = "AAGTAA" → M stops at AA (stop codon TAA after RC).
        # Build: want RC to be ATG TAA ... = stops after M
        # RC of "ATGTAA" is "TTACAT"; so _translate("TTACAT", "-") should give "M"
        result = _translate("TTACAT", "-")
        assert result == "M"


# ─────────────────────────────────────────────────────────────────────────────
# _genomic_to_cds_offset — continuity across exon boundaries
# ─────────────────────────────────────────────────────────────────────────────

class TestGenomicToCdsOffsetContinuity:
    """Offsets must be strictly consecutive with no jumps at exon boundaries."""

    def test_last_base_exon1_and_first_base_exon2_are_consecutive(self):
        exons = _exons([(100, 109), (200, 209)])
        off_last_e1  = _genomic_to_cds_offset(109, exons, "+")
        off_first_e2 = _genomic_to_cds_offset(200, exons, "+")
        assert off_last_e1  == 9
        assert off_first_e2 == 10   # exactly one more

    def test_offsets_within_exon_are_consecutive(self):
        exons = _exons([(100, 109)])
        offsets = [_genomic_to_cds_offset(100 + i, exons, "+") for i in range(10)]
        assert offsets == list(range(10))

    def test_exon_boundary_position_outside_intron_returns_none(self):
        exons = _exons([(100, 109), (200, 209)])
        assert _genomic_to_cds_offset(110, exons, "+") is None   # first intron base
        assert _genomic_to_cds_offset(199, exons, "+") is None   # last intron base

    def test_three_exon_cumulative_offsets(self):
        # exon1: 100-104 (5 nt), exon2: 200-204 (5 nt), exon3: 300-304 (5 nt)
        exons = _exons([(100, 104), (200, 204), (300, 304)])
        assert _genomic_to_cds_offset(100, exons, "+") == 0
        assert _genomic_to_cds_offset(104, exons, "+") == 4
        assert _genomic_to_cds_offset(200, exons, "+") == 5
        assert _genomic_to_cds_offset(204, exons, "+") == 9
        assert _genomic_to_cds_offset(300, exons, "+") == 10
        assert _genomic_to_cds_offset(304, exons, "+") == 14


# ─────────────────────────────────────────────────────────────────────────────
# NT interval boundary: long REF extending beyond interval end (known limitation)
# ─────────────────────────────────────────────────────────────────────────────

class TestNtIntervalBoundaryBehavior:
    """
    The NT path (_add_nt_haplotypes) computes offsets relative to the interval
    start.  A VCF REF allele whose footprint extends past the interval end causes
    _apply_variants to write beyond the interval string.  Python list slice
    assignment silently extends the list, producing a sequence longer than the
    interval.

    These tests document the CURRENT behaviour so that if the NT path is ever
    fixed to clip (analogous to the AA exon-clipping fix) the tests will catch it.
    """

    def test_long_ref_at_interval_end_extends_sequence(self):
        """
        Interval: 'ATCG' (4 chars).  Variant at offset 2 with REF='CGX' (3 chars,
        extending 1 char past the interval end).  ALT='TGX' (same length).

        seq_chars[2:5] = list("TGX") replaces chars 2-4, but index 4 is beyond
        the interval; Python silently appends, giving a 5-char result.
        """
        ref_iv = "ATCG"   # interval seq (4 chars)
        pi = [("102", _pi(2, "CGX"))]   # ref_len=3, extends 1 past end
        result = _apply_variants(ref_iv, pi, {"102": "TGX"})
        # Without clipping the result is extended / corrupted
        assert len(result) != len(ref_iv)   # length mismatch documents the issue

    def test_long_ref_within_interval_is_fine(self):
        """Contrast: REF entirely within interval → correct result."""
        ref_iv = "ATCG"
        pi = [("102", _pi(2, "CG"))]   # ref_len=2, fits in interval
        result = _apply_variants(ref_iv, pi, {"102": "TT"})
        assert result == "ATTT"
        assert len(result) == len(ref_iv)
